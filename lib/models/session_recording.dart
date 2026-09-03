import 'dart:convert';

/// Represents a complete recorded LiveKit session run.
class SessionRecording {
  final String id;
  final String? roomName;
  final String? roomSid;
  final DateTime startedAt;
  final DateTime? endedAt;
  final int? durationMs;
  final SessionLocalParticipant? localParticipant;
  final List<SessionParticipant> participants;
  final List<TranscriptTurn> transcripts;
  final List<TurnTakingMetrics> turnTakingMetrics;
  final List<AgentStateTransition> agentStateTransitions;
  final List<SessionEventLog> rawEvents;
  final SessionSummary summary;

  SessionRecording({
    required this.id,
    this.roomName,
    this.roomSid,
    required this.startedAt,
    this.endedAt,
    this.durationMs,
    this.localParticipant,
    List<SessionParticipant>? participants,
    List<TranscriptTurn>? transcripts,
    List<TurnTakingMetrics>? turnTakingMetrics,
    List<AgentStateTransition>? agentStateTransitions,
    List<SessionEventLog>? rawEvents,
    SessionSummary? summary,
  })  : participants = participants ?? [],
        transcripts = transcripts ?? [],
        turnTakingMetrics = turnTakingMetrics ?? [],
        agentStateTransitions = agentStateTransitions ?? [],
        rawEvents = rawEvents ?? [],
        summary = summary ?? SessionSummary.empty();

  Map<String, dynamic> toJson() => {
        'sessionId': id,
        'roomName': roomName,
        'roomSid': roomSid,
        'startedAt': startedAt.toIso8601String(),
        'endedAt': endedAt?.toIso8601String(),
        'durationMs': durationMs,
        'localParticipant': localParticipant?.toJson(),
        'participants': participants.map((p) => p.toJson()).toList(),
        'summary': summary.toJson(),
        'transcripts': transcripts.map((t) => t.toJson()).toList(),
        'turnTakingMetrics': turnTakingMetrics.map((m) => m.toJson()).toList(),
        'agentStateTransitions': agentStateTransitions.map((s) => s.toJson()).toList(),
        'rawEvents': rawEvents.map((e) => e.toJson()).toList(),
      };

  String toJsonString({bool pretty = true}) {
    final encoder = pretty ? const JsonEncoder.withIndent('  ') : const JsonEncoder();
    return encoder.convert(toJson());
  }

  factory SessionRecording.fromJson(Map<String, dynamic> json) => SessionRecording(
        id: json['sessionId'] as String? ?? '',
        roomName: json['roomName'] as String?,
        roomSid: json['roomSid'] as String?,
        startedAt: DateTime.parse(json['startedAt'] as String),
        endedAt: json['endedAt'] != null ? DateTime.parse(json['endedAt'] as String) : null,
        durationMs: json['durationMs'] as int?,
        localParticipant: json['localParticipant'] != null
            ? SessionLocalParticipant.fromJson(json['localParticipant'] as Map<String, dynamic>)
            : null,
        participants: (json['participants'] as List<dynamic>?)
                ?.map((e) => SessionParticipant.fromJson(e as Map<String, dynamic>))
                .toList() ??
            [],
        transcripts: (json['transcripts'] as List<dynamic>?)
                ?.map((e) => TranscriptTurn.fromJson(e as Map<String, dynamic>))
                .toList() ??
            [],
        turnTakingMetrics: (json['turnTakingMetrics'] as List<dynamic>?)
                ?.map((e) => TurnTakingMetrics.fromJson(e as Map<String, dynamic>))
                .toList() ??
            [],
        agentStateTransitions: (json['agentStateTransitions'] as List<dynamic>?)
                ?.map((e) => AgentStateTransition.fromJson(e as Map<String, dynamic>))
                .toList() ??
            [],
        rawEvents: (json['rawEvents'] as List<dynamic>?)
                ?.map((e) => SessionEventLog.fromJson(e as Map<String, dynamic>))
                .toList() ??
            [],
        summary: json['summary'] != null
            ? SessionSummary.fromJson(json['summary'] as Map<String, dynamic>)
            : SessionSummary.empty(),
      );
}

/// Identifies a speaker (user or bot/agent).
class SessionSpeaker {
  final String identity;
  final String? name;
  final String? sid;
  final bool isAgent;
  final String? kind;
  final Map<String, String>? attributes;

  const SessionSpeaker({
    required this.identity,
    this.name,
    this.sid,
    this.isAgent = false,
    this.kind,
    this.attributes,
  });

  Map<String, dynamic> toJson() => {
        'identity': identity,
        if (name != null) 'name': name,
        if (sid != null) 'sid': sid,
        'isAgent': isAgent,
        if (kind != null) 'kind': kind,
        if (attributes != null && attributes!.isNotEmpty) 'attributes': attributes,
      };

  factory SessionSpeaker.fromJson(Map<String, dynamic> json) => SessionSpeaker(
        identity: json['identity'] as String? ?? '',
        name: json['name'] as String?,
        sid: json['sid'] as String?,
        isAgent: json['isAgent'] as bool? ?? false,
        kind: json['kind'] as String?,
        attributes: (json['attributes'] as Map<String, dynamic>?)?.map(
          (k, v) => MapEntry(k, v.toString()),
        ),
      );
}

/// Information about an individual word or chunk timing within a turn.
class WordTiming {
  final String word;
  final DateTime timestamp;
  final int latencyFromPreviousWordMs;
  final int latencyFromTurnStartMs;

  const WordTiming({
    required this.word,
    required this.timestamp,
    required this.latencyFromPreviousWordMs,
    required this.latencyFromTurnStartMs,
  });

  Map<String, dynamic> toJson() => {
        'word': word,
        'timestamp': timestamp.toIso8601String(),
        'latencyFromPreviousWordMs': latencyFromPreviousWordMs,
        'latencyFromTurnStartMs': latencyFromTurnStartMs,
      };

  factory WordTiming.fromJson(Map<String, dynamic> json) => WordTiming(
        word: json['word'] as String? ?? '',
        timestamp: DateTime.parse(json['timestamp'] as String),
        latencyFromPreviousWordMs: json['latencyFromPreviousWordMs'] as int? ?? 0,
        latencyFromTurnStartMs: json['latencyFromTurnStartMs'] as int? ?? 0,
      );
}

/// A single conversational turn / line of words spoken by a participant.
class TranscriptTurn {
  final int turnIndex;
  final String id;
  final String role; // 'agent' | 'user' | 'system'
  final SessionSpeaker speaker;
  final String text;
  final bool isFinal;
  final DateTime startTimestamp;
  final DateTime? endTimestamp;
  final int durationMs;
  final List<WordTiming> words;
  final List<int> interWordLatenciesMs;
  final double? averageInterWordLatencyMs;

  const TranscriptTurn({
    required this.turnIndex,
    required this.id,
    required this.role,
    required this.speaker,
    required this.text,
    required this.isFinal,
    required this.startTimestamp,
    this.endTimestamp,
    required this.durationMs,
    required this.words,
    required this.interWordLatenciesMs,
    this.averageInterWordLatencyMs,
  });

  Map<String, dynamic> toJson() => {
        'turnIndex': turnIndex,
        'id': id,
        'role': role,
        'speaker': speaker.toJson(),
        'text': text,
        'isFinal': isFinal,
        'startTimestamp': startTimestamp.toIso8601String(),
        'endTimestamp': endTimestamp?.toIso8601String(),
        'durationMs': durationMs,
        'words': words.map((w) => w.toJson()).toList(),
        'interWordLatenciesMs': interWordLatenciesMs,
        'averageInterWordLatencyMs': averageInterWordLatencyMs,
      };

  factory TranscriptTurn.fromJson(Map<String, dynamic> json) => TranscriptTurn(
        turnIndex: json['turnIndex'] as int? ?? 0,
        id: json['id'] as String? ?? '',
        role: json['role'] as String? ?? 'agent',
        speaker: SessionSpeaker.fromJson(json['speaker'] as Map<String, dynamic>),
        text: json['text'] as String? ?? '',
        isFinal: json['isFinal'] as bool? ?? true,
        startTimestamp: DateTime.parse(json['startTimestamp'] as String),
        endTimestamp: json['endTimestamp'] != null ? DateTime.parse(json['endTimestamp'] as String) : null,
        durationMs: json['durationMs'] as int? ?? 0,
        words: (json['words'] as List<dynamic>?)?.map((w) => WordTiming.fromJson(w as Map<String, dynamic>)).toList() ??
            [],
        interWordLatenciesMs:
            (json['interWordLatenciesMs'] as List<dynamic>?)?.map((e) => (e as num).toInt()).toList() ?? [],
        averageInterWordLatencyMs: (json['averageInterWordLatencyMs'] as num?)?.toDouble(),
      );
}

/// Timestamps associated with a turn-taking exchange.
class TurnTakingTimestamps {
  final DateTime? userInputEndedAt;
  final DateTime? agentProcessingStartedAt;
  final DateTime? agentStartedSpeakingAt;
  final DateTime? agentFinishedSpeakingAt;

  const TurnTakingTimestamps({
    this.userInputEndedAt,
    this.agentProcessingStartedAt,
    this.agentStartedSpeakingAt,
    this.agentFinishedSpeakingAt,
  });

  Map<String, dynamic> toJson() => {
        if (userInputEndedAt != null) 'userInputEndedAt': userInputEndedAt!.toIso8601String(),
        if (agentProcessingStartedAt != null) 'agentProcessingStartedAt': agentProcessingStartedAt!.toIso8601String(),
        if (agentStartedSpeakingAt != null) 'agentStartedSpeakingAt': agentStartedSpeakingAt!.toIso8601String(),
        if (agentFinishedSpeakingAt != null) 'agentFinishedSpeakingAt': agentFinishedSpeakingAt!.toIso8601String(),
      };

  factory TurnTakingTimestamps.fromJson(Map<String, dynamic> json) => TurnTakingTimestamps(
        userInputEndedAt: json['userInputEndedAt'] != null ? DateTime.parse(json['userInputEndedAt'] as String) : null,
        agentProcessingStartedAt: json['agentProcessingStartedAt'] != null
            ? DateTime.parse(json['agentProcessingStartedAt'] as String)
            : null,
        agentStartedSpeakingAt:
            json['agentStartedSpeakingAt'] != null ? DateTime.parse(json['agentStartedSpeakingAt'] as String) : null,
        agentFinishedSpeakingAt:
            json['agentFinishedSpeakingAt'] != null ? DateTime.parse(json['agentFinishedSpeakingAt'] as String) : null,
      );
}

/// Latency calculations for a turn-taking exchange.
class TurnTakingLatencies {
  final int? turnTakingLatencyMs;
  final int? processingLatencyMs;
  final int? agentSpeakingDurationMs;

  const TurnTakingLatencies({
    this.turnTakingLatencyMs,
    this.processingLatencyMs,
    this.agentSpeakingDurationMs,
  });

  Map<String, dynamic> toJson() => {
        'turnTakingLatencyMs': turnTakingLatencyMs,
        'processingLatencyMs': processingLatencyMs,
        'agentSpeakingDurationMs': agentSpeakingDurationMs,
      };

  factory TurnTakingLatencies.fromJson(Map<String, dynamic> json) => TurnTakingLatencies(
        turnTakingLatencyMs: json['turnTakingLatencyMs'] as int?,
        processingLatencyMs: json['processingLatencyMs'] as int?,
        agentSpeakingDurationMs: json['agentSpeakingDurationMs'] as int?,
      );
}

/// Metrics measuring latency and timing during a user -> agent turn transition.
class TurnTakingMetrics {
  final int turnIndex;
  final String? userTurnId;
  final String? agentTurnId;
  final SessionSpeaker? agentSpeaker;
  final TurnTakingTimestamps timestamps;
  final TurnTakingLatencies latencies;

  const TurnTakingMetrics({
    required this.turnIndex,
    this.userTurnId,
    this.agentTurnId,
    this.agentSpeaker,
    required this.timestamps,
    required this.latencies,
  });

  Map<String, dynamic> toJson() => {
        'turnIndex': turnIndex,
        'userTurnId': userTurnId,
        'agentTurnId': agentTurnId,
        'agentSpeaker': agentSpeaker?.toJson(),
        'timestamps': timestamps.toJson(),
        'latencies': latencies.toJson(),
      };

  factory TurnTakingMetrics.fromJson(Map<String, dynamic> json) => TurnTakingMetrics(
        turnIndex: json['turnIndex'] as int? ?? 0,
        userTurnId: json['userTurnId'] as String?,
        agentTurnId: json['agentTurnId'] as String?,
        agentSpeaker:
            json['agentSpeaker'] != null ? SessionSpeaker.fromJson(json['agentSpeaker'] as Map<String, dynamic>) : null,
        timestamps: TurnTakingTimestamps.fromJson(json['timestamps'] as Map<String, dynamic>? ?? {}),
        latencies: TurnTakingLatencies.fromJson(json['latencies'] as Map<String, dynamic>? ?? {}),
      );
}

/// Represents an agent conversational state transition (idle, listening, thinking, speaking).
class AgentStateTransition {
  final DateTime timestamp;
  final String participantIdentity;
  final String? fromState;
  final String toState;
  final int? durationInPreviousStateMs;

  const AgentStateTransition({
    required this.timestamp,
    required this.participantIdentity,
    this.fromState,
    required this.toState,
    this.durationInPreviousStateMs,
  });

  Map<String, dynamic> toJson() => {
        'timestamp': timestamp.toIso8601String(),
        'participantIdentity': participantIdentity,
        'fromState': fromState,
        'toState': toState,
        'durationInPreviousStateMs': durationInPreviousStateMs,
      };

  factory AgentStateTransition.fromJson(Map<String, dynamic> json) => AgentStateTransition(
        timestamp: DateTime.parse(json['timestamp'] as String),
        participantIdentity: json['participantIdentity'] as String? ?? '',
        fromState: json['fromState'] as String?,
        toState: json['toState'] as String? ?? '',
        durationInPreviousStateMs: json['durationInPreviousStateMs'] as int?,
      );
}

/// Generic log entry for a timeline event.
class SessionEventLog {
  final DateTime timestamp;
  final String eventType;
  final Map<String, dynamic> data;

  const SessionEventLog({
    required this.timestamp,
    required this.eventType,
    required this.data,
  });

  Map<String, dynamic> toJson() => {
        'timestamp': timestamp.toIso8601String(),
        'eventType': eventType,
        'data': data,
      };

  factory SessionEventLog.fromJson(Map<String, dynamic> json) => SessionEventLog(
        timestamp: DateTime.parse(json['timestamp'] as String),
        eventType: json['eventType'] as String? ?? '',
        data: json['data'] as Map<String, dynamic>? ?? {},
      );
}

/// Summary metrics for the entire session run.
class SessionSummary {
  final int totalTurns;
  final int userTurns;
  final int agentTurns;
  final int totalWords;
  final double? averageTurnTakingLatencyMs;
  final double? averageProcessingLatencyMs;
  final double? averageInterWordLatencyMs;

  const SessionSummary({
    required this.totalTurns,
    required this.userTurns,
    required this.agentTurns,
    required this.totalWords,
    this.averageTurnTakingLatencyMs,
    this.averageProcessingLatencyMs,
    this.averageInterWordLatencyMs,
  });

  factory SessionSummary.empty() => const SessionSummary(
        totalTurns: 0,
        userTurns: 0,
        agentTurns: 0,
        totalWords: 0,
      );

  Map<String, dynamic> toJson() => {
        'totalTurns': totalTurns,
        'userTurns': userTurns,
        'agentTurns': agentTurns,
        'totalWords': totalWords,
        'averageTurnTakingLatencyMs': averageTurnTakingLatencyMs,
        'averageProcessingLatencyMs': averageProcessingLatencyMs,
        'averageInterWordLatencyMs': averageInterWordLatencyMs,
      };

  factory SessionSummary.fromJson(Map<String, dynamic> json) => SessionSummary(
        totalTurns: json['totalTurns'] as int? ?? 0,
        userTurns: json['userTurns'] as int? ?? 0,
        agentTurns: json['agentTurns'] as int? ?? 0,
        totalWords: json['totalWords'] as int? ?? 0,
        averageTurnTakingLatencyMs: (json['averageTurnTakingLatencyMs'] as num?)?.toDouble(),
        averageProcessingLatencyMs: (json['averageProcessingLatencyMs'] as num?)?.toDouble(),
        averageInterWordLatencyMs: (json['averageInterWordLatencyMs'] as num?)?.toDouble(),
      );
}

/// Local participant record.
class SessionLocalParticipant {
  final String identity;
  final String? sid;
  final String? name;

  const SessionLocalParticipant({
    required this.identity,
    this.sid,
    this.name,
  });

  Map<String, dynamic> toJson() => {
        'identity': identity,
        if (sid != null) 'sid': sid,
        if (name != null) 'name': name,
      };

  factory SessionLocalParticipant.fromJson(Map<String, dynamic> json) => SessionLocalParticipant(
        identity: json['identity'] as String? ?? '',
        sid: json['sid'] as String?,
        name: json['name'] as String?,
      );
}

/// Participant record in the session.
class SessionParticipant {
  final String identity;
  final String? sid;
  final String? name;
  final bool isAgent;
  final String? kind;
  final Map<String, String>? attributes;
  final DateTime? joinedAt;
  final DateTime? leftAt;

  const SessionParticipant({
    required this.identity,
    this.sid,
    this.name,
    this.isAgent = false,
    this.kind,
    this.attributes,
    this.joinedAt,
    this.leftAt,
  });

  Map<String, dynamic> toJson() => {
        'identity': identity,
        if (sid != null) 'sid': sid,
        if (name != null) 'name': name,
        'isAgent': isAgent,
        if (kind != null) 'kind': kind,
        if (attributes != null && attributes!.isNotEmpty) 'attributes': attributes,
        if (joinedAt != null) 'joinedAt': joinedAt!.toIso8601String(),
        if (leftAt != null) 'leftAt': leftAt!.toIso8601String(),
      };

  factory SessionParticipant.fromJson(Map<String, dynamic> json) => SessionParticipant(
        identity: json['identity'] as String? ?? '',
        sid: json['sid'] as String?,
        name: json['name'] as String?,
        isAgent: json['isAgent'] as bool? ?? false,
        kind: json['kind'] as String?,
        attributes: (json['attributes'] as Map<String, dynamic>?)?.map(
          (k, v) => MapEntry(k, v.toString()),
        ),
        joinedAt: json['joinedAt'] != null ? DateTime.parse(json['joinedAt'] as String) : null,
        leftAt: json['leftAt'] != null ? DateTime.parse(json['leftAt'] as String) : null,
      );
}
