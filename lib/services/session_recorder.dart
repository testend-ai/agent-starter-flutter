import 'dart:async';
import 'dart:collection';
import 'dart:convert';
import 'dart:io';

import 'package:livekit_client/livekit_client.dart' as sdk;
import 'package:logging/logging.dart';
import 'package:path_provider/path_provider.dart';
import 'package:uuid/uuid.dart';

import '../exts.dart';
import '../models/session_recording.dart';

const String _agentStateKey = 'lk.agent.state';

/// Service responsible for recording all LiveKit session data, transcripts,
/// bot identities, inter-word latencies, agent state transitions, turn-taking
/// latency, and writing the final report to a JSON file every single run.
class SessionRecorder {
  static final _logger = Logger('SessionRecorder');
  static const _uuid = Uuid();

  final sdk.Room room;

  SessionRecording? _currentRecording;
  SessionRecording? get currentRecording => _currentRecording;

  SessionRecording? _lastSavedRecording;
  SessionRecording? get lastSavedRecording => _lastSavedRecording;

  String? _lastSavedFilePath;
  String? get lastSavedFilePath => _lastSavedFilePath;

  String? _customSaveDirectory;
  set customSaveDirectory(String? dir) => _customSaveDirectory = dir;

  sdk.EventsListener<sdk.RoomEvent>? _roomListener;
  final Map<String, sdk.EventsListener<sdk.ParticipantEvent>> _participantListeners = HashMap();

  // Turn-taking tracking state
  final Map<String, _ActiveTurnBuilder> _activeTurns = HashMap();
  int _turnIndexCounter = 0;

  DateTime? _lastUserInputEndedAt;
  String? _lastUserTurnId;

  DateTime? _agentProcessingStartedAt;
  DateTime? _agentStartedSpeakingAt;
  DateTime? _agentFinishedSpeakingAt;
  String? _currentAgentTurnId;
  SessionSpeaker? _currentAgentSpeaker;

  final Map<String, _ParticipantTracker> _participants = HashMap();
  bool _isSessionActive = false;
  bool _isFinalized = false;

  SessionRecorder({
    required this.room,
    String? customSaveDirectory,
  }) : _customSaveDirectory = customSaveDirectory {
    _observeRoom();
  }

  /// Sets the associated [sdk.Session] instance (if needed for coordination).
  void attachSession(sdk.Session session) {
    // Attached session
  }

  /// Starts recording a new session run.
  void startNewSession({String? roomName, String? roomSid}) {
    final sessionId = _uuid.v4();
    final now = DateTime.now();

    _turnIndexCounter = 0;
    _activeTurns.clear();
    _participants.clear();
    _lastUserInputEndedAt = null;
    _lastUserTurnId = null;
    _agentProcessingStartedAt = null;
    _agentStartedSpeakingAt = null;
    _agentFinishedSpeakingAt = null;
    _currentAgentTurnId = null;
    _currentAgentSpeaker = null;
    _isFinalized = false;
    _isSessionActive = true;

    final localParticipant = room.localParticipant;
    final sessionLocal = localParticipant != null
        ? SessionLocalParticipant(
            identity: localParticipant.identity,
            sid: localParticipant.sid,
            name: localParticipant.name,
          )
        : null;

    _currentRecording = SessionRecording(
      id: sessionId,
      roomName: roomName ?? room.name,
      roomSid: roomSid,
      startedAt: now,
      localParticipant: sessionLocal,
    );

    _logEvent('session_started', {
      'sessionId': sessionId,
      'roomName': _currentRecording?.roomName,
      'roomSid': _currentRecording?.roomSid,
    });

    _recordExistingParticipants();
  }

  void _recordExistingParticipants() {
    final local = room.localParticipant;
    if (local != null) {
      _trackParticipant(local);
    }
    for (final participant in room.remoteParticipants.values) {
      _trackParticipant(participant);
    }
  }

  void _trackParticipant(sdk.Participant participant) {
    final tracker = _participants.putIfAbsent(
      participant.identity,
      () => _ParticipantTracker(
        identity: participant.identity,
        sid: participant.sid,
        name: participant.name,
        isAgent: participant.isAgent,
        kind: participant.kind.name,
        attributes: Map.from(participant.attributes),
        joinedAt: DateTime.now(),
      ),
    );

    tracker.sid = participant.sid;
    tracker.name = participant.name;
    tracker.isAgent = participant.isAgent;
    tracker.kind = participant.kind.name;
    tracker.attributes = Map.from(participant.attributes);

    _listenToParticipant(participant);
  }

  void _listenToParticipant(sdk.Participant participant) {
    if (_participantListeners.containsKey(participant.identity)) {
      return;
    }

    final listener = participant.createListener();
    listener.listen((event) {
      if (event is sdk.SpeakingChangedEvent) {
        _handleSpeakingChanged(event.participant, event.speaking);
      } else if (event is sdk.ParticipantAttributesChanged) {
        _trackParticipant(event.participant);
        _handleParticipantAttributesChanged(event.participant, event.attributes);
      }
    });
    _participantListeners[participant.identity] = listener;
  }

  void _observeRoom() {
    final listener = room.createListener();
    listener.listen((event) {
      _handleRoomEvent(event);
    });
    _roomListener = listener;
  }

  void _handleRoomEvent(sdk.RoomEvent event) {
    if (!_isSessionActive && event is sdk.RoomConnectedEvent) {
      startNewSession(roomName: event.room.name);
    }

    if (!_isSessionActive && _currentRecording == null) {
      return;
    }

    if (event is sdk.RoomConnectedEvent) {
      _logEvent('room_connected', {
        'roomName': event.room.name,
      });
      _recordExistingParticipants();
    } else if (event is sdk.RoomDisconnectedEvent) {
      _logEvent('room_disconnected', {
        'reason': event.reason?.toString(),
      });
      unawaited(finalizeAndSave());
    } else if (event is sdk.ParticipantConnectedEvent) {
      _trackParticipant(event.participant);
      _logEvent('participant_connected', {
        'identity': event.participant.identity,
        'name': event.participant.name,
        'isAgent': event.participant.isAgent,
      });
    } else if (event is sdk.ParticipantDisconnectedEvent) {
      final tracker = _participants[event.participant.identity];
      tracker?.leftAt = DateTime.now();
      final listener = _participantListeners.remove(event.participant.identity);
      unawaited(listener?.dispose());
      _logEvent('participant_disconnected', {
        'identity': event.participant.identity,
      });
    } else if (event is sdk.ParticipantAttributesChanged) {
      _trackParticipant(event.participant);
      _handleParticipantAttributesChanged(event.participant, event.attributes);
    } else if (event is sdk.ActiveSpeakersChangedEvent) {
      _handleActiveSpeakersChanged(event.speakers);
    } else if (event is sdk.TranscriptionEvent) {
      _handleServerTranscriptionEvent(event);
    }
  }

  void _handleActiveSpeakersChanged(List<sdk.Participant> speakers) {
    final speakerIdentities = speakers.map((s) => s.identity).toSet();
    final localIdentity = room.localParticipant?.identity;

    if (localIdentity != null && !speakerIdentities.contains(localIdentity)) {
      _lastUserInputEndedAt = DateTime.now();
    }

    for (final speaker in speakers) {
      if (speaker.isAgent) {
        _agentStartedSpeakingAt ??= DateTime.now();
      }
    }
  }

  void _handleSpeakingChanged(sdk.Participant participant, bool speaking) {
    final now = DateTime.now();
    final isLocal = participant is sdk.LocalParticipant || participant.identity == room.localParticipant?.identity;

    _logEvent('speaking_changed', {
      'identity': participant.identity,
      'isLocal': isLocal,
      'speaking': speaking,
    });

    if (isLocal) {
      if (!speaking) {
        _lastUserInputEndedAt = now;
      }
    } else {
      if (speaking) {
        _agentStartedSpeakingAt ??= now;
      } else {
        if (_agentStartedSpeakingAt != null) {
          _agentFinishedSpeakingAt = now;
        }
      }
    }
  }

  void _handleParticipantAttributesChanged(sdk.Participant participant, Map<String, String> attributes) {
    final now = DateTime.now();
    final tracker = _participants[participant.identity];
    final oldState = tracker?.lastAgentState;
    final newState = participant.agentState?.name ?? attributes[_agentStateKey];

    if (newState != null && newState != oldState) {
      int? durationInPrev;
      if (tracker?.lastStateChangeTime != null) {
        durationInPrev = now.difference(tracker!.lastStateChangeTime!).inMilliseconds;
      }

      tracker?.lastAgentState = newState;
      tracker?.lastStateChangeTime = now;

      final transition = AgentStateTransition(
        timestamp: now,
        participantIdentity: participant.identity,
        fromState: oldState,
        toState: newState,
        durationInPreviousStateMs: durationInPrev,
      );

      _currentRecording?.agentStateTransitions.add(transition);

      _logEvent('agent_state_changed', {
        'identity': participant.identity,
        'fromState': oldState,
        'toState': newState,
        'durationInPreviousStateMs': durationInPrev,
      });

      // Turntaking state tracking
      if (newState == 'thinking') {
        _agentProcessingStartedAt = now;
      } else if (newState == 'speaking') {
        _agentStartedSpeakingAt = now;
        _currentAgentSpeaker = _resolveSpeaker(participant.identity);
      } else if (newState == 'listening' || newState == 'idle') {
        if (_agentStartedSpeakingAt != null) {
          _agentFinishedSpeakingAt = now;
          _recordTurnTakingMetricsIfReady();
        }
      }
    }
  }

  void _handleServerTranscriptionEvent(sdk.TranscriptionEvent event) {
    final now = DateTime.now();
    for (final segment in event.segments) {
      onTranscriptionChunk(
        chunkText: segment.text,
        segmentId: segment.id,
        participantIdentity: event.participant.identity,
        isFinal: segment.isFinal,
        timestamp: now,
      );
    }
  }

  /// Called when a transcription chunk arrives via text stream or STT receiver.
  void onTranscriptionChunk({
    required String chunkText,
    required String segmentId,
    required String participantIdentity,
    required bool isFinal,
    required DateTime timestamp,
  }) {
    if (!_isSessionActive && _currentRecording == null) {
      startNewSession();
    }

    final isLocal = room.localParticipant?.identity == participantIdentity;
    final role = isLocal ? 'user' : 'agent';
    final speaker = _resolveSpeaker(participantIdentity);

    var turnBuilder = _activeTurns[segmentId];
    if (turnBuilder == null) {
      turnBuilder = _ActiveTurnBuilder(
        turnIndex: _turnIndexCounter++,
        segmentId: segmentId,
        role: role,
        speaker: speaker,
        startTimestamp: timestamp,
      );
      _activeTurns[segmentId] = turnBuilder;

      if (!isLocal) {
        _currentAgentTurnId = segmentId;
        _currentAgentSpeaker = speaker;
        _agentStartedSpeakingAt ??= timestamp;
      } else {
        _lastUserTurnId = segmentId;
      }
    }

    turnBuilder.appendChunk(chunkText, timestamp);

    if (isFinal) {
      turnBuilder.isFinal = true;
      turnBuilder.endTimestamp = timestamp;
      final completedTurn = turnBuilder.build();
      _currentRecording?.transcripts.add(completedTurn);
      _activeTurns.remove(segmentId);

      if (isLocal) {
        _lastUserInputEndedAt = timestamp;
        _lastUserTurnId = completedTurn.id;
      } else {
        _currentAgentTurnId = completedTurn.id;
        _agentFinishedSpeakingAt = timestamp;
        _recordTurnTakingMetricsIfReady();
      }

      _logEvent('transcript_turn_completed', {
        'id': completedTurn.id,
        'role': completedTurn.role,
        'speaker': completedTurn.speaker.identity,
        'wordCount': completedTurn.words.length,
        'durationMs': completedTurn.durationMs,
        'text': completedTurn.text,
      });
    }
  }

  /// Records user input sent via text chat.
  void onUserTextMessage({
    required String id,
    required String text,
    required DateTime timestamp,
  }) {
    if (!_isSessionActive && _currentRecording == null) {
      startNewSession();
    }

    final localIdentity = room.localParticipant?.identity ?? 'user';
    final speaker = _resolveSpeaker(localIdentity);

    final turnBuilder = _ActiveTurnBuilder(
      turnIndex: _turnIndexCounter++,
      segmentId: id,
      role: 'user',
      speaker: speaker,
      startTimestamp: timestamp,
    );
    turnBuilder.appendChunk(text, timestamp);
    turnBuilder.isFinal = true;
    turnBuilder.endTimestamp = timestamp;

    final completedTurn = turnBuilder.build();
    _currentRecording?.transcripts.add(completedTurn);

    _lastUserInputEndedAt = timestamp;
    _lastUserTurnId = id;

    _logEvent('user_text_message', {
      'id': id,
      'text': text,
      'timestamp': timestamp.toIso8601String(),
    });
  }

  void _recordTurnTakingMetricsIfReady() {
    if (_lastUserInputEndedAt == null || _agentStartedSpeakingAt == null) {
      return;
    }

    final turnTakingLatencyMs = _agentStartedSpeakingAt!.difference(_lastUserInputEndedAt!).inMilliseconds;
    final processingLatencyMs = _agentProcessingStartedAt?.difference(_agentStartedSpeakingAt!).inMilliseconds.abs();
    final agentSpeakingDurationMs = _agentFinishedSpeakingAt?.difference(_agentStartedSpeakingAt!).inMilliseconds.abs();

    final metrics = TurnTakingMetrics(
      turnIndex: _currentRecording?.turnTakingMetrics.length ?? 0,
      userTurnId: _lastUserTurnId,
      agentTurnId: _currentAgentTurnId,
      agentSpeaker: _currentAgentSpeaker,
      timestamps: TurnTakingTimestamps(
        userInputEndedAt: _lastUserInputEndedAt,
        agentProcessingStartedAt: _agentProcessingStartedAt,
        agentStartedSpeakingAt: _agentStartedSpeakingAt,
        agentFinishedSpeakingAt: _agentFinishedSpeakingAt,
      ),
      latencies: TurnTakingLatencies(
        turnTakingLatencyMs: turnTakingLatencyMs,
        processingLatencyMs: processingLatencyMs,
        agentSpeakingDurationMs: agentSpeakingDurationMs,
      ),
    );

    _currentRecording?.turnTakingMetrics.add(metrics);

    _logEvent('turn_taking_measured', {
      'turnIndex': metrics.turnIndex,
      'turnTakingLatencyMs': turnTakingLatencyMs,
      'processingLatencyMs': processingLatencyMs,
      'agentSpeakingDurationMs': agentSpeakingDurationMs,
    });

    // Reset turn-taking timestamps for the next exchange
    _lastUserInputEndedAt = null;
    _agentProcessingStartedAt = null;
    _agentStartedSpeakingAt = null;
    _agentFinishedSpeakingAt = null;
  }

  SessionSpeaker _resolveSpeaker(String identity) {
    final local = room.localParticipant;
    final sdk.Participant? participant =
        room.remoteParticipants[identity] ?? (local?.identity == identity ? local : null);

    if (participant != null) {
      return SessionSpeaker(
        identity: participant.identity,
        name: participant.name,
        sid: participant.sid,
        isAgent: participant.isAgent,
        kind: participant.kind.name,
        attributes: Map.from(participant.attributes),
      );
    }

    final tracker = _participants[identity];
    if (tracker != null) {
      return SessionSpeaker(
        identity: tracker.identity,
        name: tracker.name,
        sid: tracker.sid,
        isAgent: tracker.isAgent,
        kind: tracker.kind,
        attributes: tracker.attributes,
      );
    }

    final isLocal = room.localParticipant?.identity == identity;
    return SessionSpeaker(
      identity: identity,
      isAgent: !isLocal,
      kind: isLocal ? 'STANDARD' : 'AGENT',
    );
  }

  void _logEvent(String eventType, Map<String, dynamic> data) {
    final entry = SessionEventLog(
      timestamp: DateTime.now(),
      eventType: eventType,
      data: data,
    );
    _currentRecording?.rawEvents.add(entry);
  }

  /// Finalizes the current session metrics and saves the recording to a JSON file.
  Future<File?> finalizeAndSave({String? customDirectoryPath}) async {
    if (_isFinalized || _currentRecording == null) {
      return _lastSavedFilePath != null ? File(_lastSavedFilePath!) : null;
    }

    _isFinalized = true;
    _isSessionActive = false;

    // Finalize any in-flight active turns
    for (final turnBuilder in _activeTurns.values) {
      turnBuilder.endTimestamp ??= DateTime.now();
      _currentRecording!.transcripts.add(turnBuilder.build());
    }
    _activeTurns.clear();

    // Check if turn taking metrics can be finalized
    _recordTurnTakingMetricsIfReady();

    final now = DateTime.now();
    final durationMs = now.difference(_currentRecording!.startedAt).inMilliseconds;

    // Collect all participants
    final participantsList = _participants.values
        .map((p) => SessionParticipant(
              identity: p.identity,
              sid: p.sid,
              name: p.name,
              isAgent: p.isAgent,
              kind: p.kind,
              attributes: p.attributes,
              joinedAt: p.joinedAt,
              leftAt: p.leftAt,
            ))
        .toList();

    // Calculate session summary
    final userTurns = _currentRecording!.transcripts.where((t) => t.role == 'user').length;
    final agentTurns = _currentRecording!.transcripts.where((t) => t.role == 'agent').length;
    final totalTurns = _currentRecording!.transcripts.length;

    int totalWords = 0;
    final allInterWordLatencies = <int>[];
    for (final turn in _currentRecording!.transcripts) {
      totalWords += turn.words.length;
      allInterWordLatencies.addAll(turn.interWordLatenciesMs);
    }

    double? avgTurnTakingLatency;
    final turnTakingLatencies =
        _currentRecording!.turnTakingMetrics.map((m) => m.latencies.turnTakingLatencyMs).whereType<int>().toList();
    if (turnTakingLatencies.isNotEmpty) {
      avgTurnTakingLatency = turnTakingLatencies.reduce((a, b) => a + b) / turnTakingLatencies.length;
    }

    double? avgProcessingLatency;
    final processingLatencies =
        _currentRecording!.turnTakingMetrics.map((m) => m.latencies.processingLatencyMs).whereType<int>().toList();
    if (processingLatencies.isNotEmpty) {
      avgProcessingLatency = processingLatencies.reduce((a, b) => a + b) / processingLatencies.length;
    }

    double? avgInterWordLatency;
    if (allInterWordLatencies.isNotEmpty) {
      avgInterWordLatency = allInterWordLatencies.reduce((a, b) => a + b) / allInterWordLatencies.length;
    }

    final summary = SessionSummary(
      totalTurns: totalTurns,
      userTurns: userTurns,
      agentTurns: agentTurns,
      totalWords: totalWords,
      averageTurnTakingLatencyMs: avgTurnTakingLatency,
      averageProcessingLatencyMs: avgProcessingLatency,
      averageInterWordLatencyMs: avgInterWordLatency,
    );

    _currentRecording = SessionRecording(
      id: _currentRecording!.id,
      roomName: _currentRecording!.roomName ?? room.name,
      roomSid: _currentRecording!.roomSid,
      startedAt: _currentRecording!.startedAt,
      endedAt: now,
      durationMs: durationMs,
      localParticipant: _currentRecording!.localParticipant,
      participants: participantsList,
      transcripts: _currentRecording!.transcripts,
      turnTakingMetrics: _currentRecording!.turnTakingMetrics,
      agentStateTransitions: _currentRecording!.agentStateTransitions,
      rawEvents: _currentRecording!.rawEvents,
      summary: summary,
    );

    _lastSavedRecording = _currentRecording;

    // Save JSON to file
    final file = await _saveRecordingToFile(
      recording: _currentRecording!,
      targetDirectory: customDirectoryPath ?? _customSaveDirectory,
    );

    if (file != null) {
      _lastSavedFilePath = file.path;
      _logger.info('Session recorded and saved to: ${file.path}');
    }

    return file;
  }

  Future<File?> _saveRecordingToFile({
    required SessionRecording recording,
    String? targetDirectory,
  }) async {
    try {
      Directory dir;
      if (targetDirectory != null && targetDirectory.isNotEmpty) {
        dir = Directory(targetDirectory);
      } else {
        try {
          final appDocDir = await getApplicationDocumentsDirectory();
          dir = Directory('${appDocDir.path}/session_records');
        } catch (_) {
          dir = Directory('session_records');
        }
      }

      if (!await dir.exists()) {
        await dir.create(recursive: true);
      }

      final formattedDate = recording.startedAt.toIso8601String().replaceAll(':', '-').replaceAll('.', '-');
      final fileId = recording.id.length >= 8 ? recording.id.substring(0, 8) : recording.id;
      final fileName = 'session_${formattedDate}_$fileId.json';
      final file = File('${dir.path}/$fileName');

      final jsonContent = recording.toJsonString(pretty: true);
      await file.writeAsString(jsonContent);

      return file;
    } catch (e, stackTrace) {
      _logger.warning('Failed to save session recording to file: $e', e, stackTrace);
      return null;
    }
  }

  /// Cleans up resources.
  Future<void> dispose() async {
    await finalizeAndSave();
    await _roomListener?.dispose();
    for (final listener in _participantListeners.values) {
      await listener.dispose();
    }
    _participantListeners.clear();
  }
}

class _ActiveTurnBuilder {
  final int turnIndex;
  final String segmentId;
  final String role;
  final SessionSpeaker speaker;
  final DateTime startTimestamp;
  DateTime? endTimestamp;
  bool isFinal = false;

  final StringBuffer _contentBuffer = StringBuffer();
  final List<WordTiming> _words = [];
  final List<int> _interWordLatencies = [];
  DateTime? _previousWordTimestamp;

  _ActiveTurnBuilder({
    required this.turnIndex,
    required this.segmentId,
    required this.role,
    required this.speaker,
    required this.startTimestamp,
  });

  void appendChunk(String chunkText, DateTime timestamp) {
    _contentBuffer.write(chunkText);

    // Extract words from the chunk
    final trimmed = chunkText.trim();
    if (trimmed.isEmpty) return;

    final wordsInChunk = trimmed.split(RegExp(r'\s+'));
    for (final word in wordsInChunk) {
      if (word.isEmpty) continue;

      final latencyFromPrev =
          _previousWordTimestamp != null ? timestamp.difference(_previousWordTimestamp!).inMilliseconds : 0;
      final latencyFromStart = timestamp.difference(startTimestamp).inMilliseconds;

      if (_words.isNotEmpty) {
        _interWordLatencies.add(latencyFromPrev);
      }

      _words.add(WordTiming(
        word: word,
        timestamp: timestamp,
        latencyFromPreviousWordMs: latencyFromPrev,
        latencyFromTurnStartMs: latencyFromStart,
      ));

      _previousWordTimestamp = timestamp;
    }
  }

  TranscriptTurn build() {
    final text = _contentBuffer.toString().trim();
    final end = endTimestamp ?? DateTime.now();
    final duration = end.difference(startTimestamp).inMilliseconds;

    double? avgInterWordLatency;
    if (_interWordLatencies.isNotEmpty) {
      avgInterWordLatency = _interWordLatencies.reduce((a, b) => a + b) / _interWordLatencies.length;
    }

    return TranscriptTurn(
      turnIndex: turnIndex,
      id: segmentId,
      role: role,
      speaker: speaker,
      text: text,
      isFinal: isFinal,
      startTimestamp: startTimestamp,
      endTimestamp: end,
      durationMs: duration,
      words: List.unmodifiable(_words),
      interWordLatenciesMs: List.unmodifiable(_interWordLatencies),
      averageInterWordLatencyMs: avgInterWordLatency,
    );
  }
}

class _ParticipantTracker {
  final String identity;
  String? sid;
  String? name;
  bool isAgent;
  String? kind;
  Map<String, String>? attributes;
  DateTime? joinedAt;
  DateTime? leftAt;

  String? lastAgentState;
  DateTime? lastStateChangeTime;

  _ParticipantTracker({
    required this.identity,
    this.sid,
    this.name,
    this.isAgent = false,
    this.kind,
    this.attributes,
    this.joinedAt,
  });
}

/// A [sdk.MessageReceiver] that receives transcription text streams, passes
/// [sdk.ReceivedMessage] objects to the LiveKit session/UI, and intercepts
/// fine-grained chunks for the [SessionRecorder].
class RecordingTranscriptionReceiver implements sdk.MessageReceiver {
  RecordingTranscriptionReceiver({
    required sdk.Room room,
    this.recorder,
    this.topic = 'lk.transcription',
    void Function(String topic, sdk.TextStreamHandler handler)? registerHandler,
    void Function(String topic)? unregisterHandler,
  })  : _room = room,
        _registerHandler = registerHandler ?? room.registerTextStreamHandler,
        _unregisterHandler = unregisterHandler ?? room.unregisterTextStreamHandler;

  final sdk.Room _room;
  final SessionRecorder? recorder;
  final String topic;
  final void Function(String, sdk.TextStreamHandler) _registerHandler;
  final void Function(String) _unregisterHandler;

  StreamController<sdk.ReceivedMessage>? _controller;
  bool _registered = false;
  bool _controllerClosed = false;

  final Map<_PartialMessageId, _PartialMessage> _partialMessages = HashMap();

  @override
  Stream<sdk.ReceivedMessage> messages() {
    if (_controller != null) {
      return _controller!.stream;
    }

    _controller = StreamController<sdk.ReceivedMessage>.broadcast(
      onListen: _registerRoomHandler,
      onCancel: _handleCancel,
    );
    _controllerClosed = false;
    return _controller!.stream;
  }

  void _registerRoomHandler() {
    if (_registered) {
      return;
    }
    _registered = true;

    _registerHandler(topic, (sdk.TextStreamReader reader, String participantIdentity) {
      reader.listen(
        (chunk) {
          final info = reader.info;
          if (info == null) {
            return;
          }

          if (chunk.content.isEmpty) {
            return;
          }

          final String text;
          try {
            text = utf8.decode(chunk.content);
          } catch (error) {
            return;
          }

          if (text.isEmpty) {
            return;
          }

          final message = _processIncoming(
            text,
            info,
            participantIdentity,
          );
          if (!_controller!.isClosed) {
            _controller!.add(message);
          }
        },
        onError: (Object error, StackTrace stackTrace) {
          if (!_controller!.isClosed) {
            _controller!.addError(error, stackTrace);
          }
        },
        onDone: () {
          final info = reader.info;
          if (info != null) {
            final segmentId = _extractSegmentId(info.attributes, info.id);
            final key = _PartialMessageId(segmentId: segmentId, participantId: participantIdentity);
            _partialMessages.remove(key);
          }
        },
        cancelOnError: true,
      );
    });
  }

  void _handleCancel() {
    if (_registered) {
      _unregisterHandler(topic);
      _registered = false;
    }
    _partialMessages.clear();
    if (_controllerClosed) {
      return;
    }
    _controllerClosed = true;
    final controller = _controller;
    _controller = null;
    if (controller != null) {
      unawaited(controller.close());
    }
  }

  sdk.ReceivedMessage _processIncoming(
    String chunk,
    sdk.TextStreamInfo info,
    String participantIdentity,
  ) {
    final segmentId = _extractSegmentId(info.attributes, info.id);
    final isFinal = _parseBool(info.attributes['lk.transcription_final']) ?? false;
    final now = DateTime.now();

    // Inform recorder of chunk arrival with timing
    recorder?.onTranscriptionChunk(
      chunkText: chunk,
      segmentId: segmentId,
      participantIdentity: participantIdentity,
      isFinal: isFinal,
      timestamp: now,
    );

    final key = _PartialMessageId(segmentId: segmentId, participantId: participantIdentity);
    final currentStreamId = info.id;
    final DateTime timestamp = DateTime.fromMillisecondsSinceEpoch(info.timestamp, isUtc: true).toLocal();

    final existing = _partialMessages[key];
    if (existing != null) {
      if (existing.streamId == currentStreamId) {
        existing.append(chunk);
      } else {
        existing.replace(chunk, currentStreamId);
      }
    } else {
      _partialMessages[key] = _PartialMessage(
        content: chunk,
        timestamp: timestamp,
        streamId: currentStreamId,
      );
      _cleanupPreviousTurn(participantIdentity, segmentId);
    }

    final currentPartial = _partialMessages[key];

    if (isFinal) {
      _partialMessages.remove(key);
    }

    final partial = isFinal ? currentPartial : _partialMessages[key];
    final displayContent = partial?.content ?? chunk;
    final displayTimestamp = partial?.timestamp ?? timestamp;
    final isLocalParticipant = _room.localParticipant?.identity == participantIdentity;

    final sdk.ReceivedMessageContent content =
        isLocalParticipant ? sdk.UserTranscript(displayContent) : sdk.AgentTranscript(displayContent);

    return sdk.ReceivedMessage(
      id: segmentId,
      timestamp: displayTimestamp,
      content: content,
    );
  }

  void _cleanupPreviousTurn(String participantId, String currentSegmentId) {
    final keysToRemove = _partialMessages.keys
        .where((key) => key.participantId == participantId && key.segmentId != currentSegmentId)
        .toList(growable: false);

    for (final key in keysToRemove) {
      _partialMessages.remove(key);
    }
  }

  String _extractSegmentId(Map<String, String> attributes, String fallback) {
    return attributes['lk.segment_id'] ?? fallback;
  }

  bool? _parseBool(String? value) {
    if (value == null) return null;
    final normalized = value.toLowerCase();
    if (normalized == 'true' || normalized == '1') return true;
    if (normalized == 'false' || normalized == '0') return false;
    return null;
  }

  @override
  Future<void> dispose() async {
    if (_registered) {
      _room.unregisterTextStreamHandler(topic);
      _registered = false;
    }
    _partialMessages.clear();
    if (!_controllerClosed) {
      _controllerClosed = true;
      final controller = _controller;
      _controller = null;
      if (controller != null) {
        await controller.close();
      }
    }
  }
}

class _PartialMessageId {
  _PartialMessageId({
    required this.segmentId,
    required this.participantId,
  });

  final String segmentId;
  final String participantId;

  @override
  bool operator ==(Object other) =>
      other is _PartialMessageId && other.segmentId == segmentId && other.participantId == participantId;

  @override
  int get hashCode => Object.hash(segmentId, participantId);
}

class _PartialMessage {
  _PartialMessage({
    required this.content,
    required this.timestamp,
    required this.streamId,
  });

  String content;
  DateTime timestamp;
  String streamId;

  void append(String chunk) {
    content += chunk;
  }

  void replace(String chunk, String newStreamId) {
    content = chunk;
    streamId = newStreamId;
  }
}
