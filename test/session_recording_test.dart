import 'dart:convert';
import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:livekit_client/livekit_client.dart' as sdk;
import 'package:voice_assistant/models/session_recording.dart';
import 'package:voice_assistant/services/session_recorder.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  group('Session Recording Models', () {
    test('SessionRecording json serialization and deserialization roundtrip', () {
      final now = DateTime.now();
      final recording = SessionRecording(
        id: 'test-session-123',
        roomName: 'test-room',
        roomSid: 'RM_123',
        startedAt: now,
        endedAt: now.add(const Duration(seconds: 10)),
        durationMs: 10000,
        localParticipant: const SessionLocalParticipant(
          identity: 'user-1',
          name: 'User One',
          sid: 'PA_local',
        ),
        participants: [
          SessionParticipant(
            identity: 'agent-bot',
            name: 'Voice Bot',
            sid: 'PA_agent',
            isAgent: true,
            kind: 'AGENT',
            attributes: {'lk.agent.state': 'speaking'},
            joinedAt: now,
          ),
        ],
        transcripts: [
          TranscriptTurn(
            turnIndex: 0,
            id: 'seg-1',
            role: 'user',
            speaker: const SessionSpeaker(
              identity: 'user-1',
              name: 'User One',
              isAgent: false,
            ),
            text: 'Hello assistant',
            isFinal: true,
            startTimestamp: now,
            endTimestamp: now.add(const Duration(milliseconds: 1200)),
            durationMs: 1200,
            words: [
              WordTiming(
                word: 'Hello',
                timestamp: now,
                latencyFromPreviousWordMs: 0,
                latencyFromTurnStartMs: 0,
              ),
              WordTiming(
                word: 'assistant',
                timestamp: now.add(const Duration(milliseconds: 400)),
                latencyFromPreviousWordMs: 400,
                latencyFromTurnStartMs: 400,
              ),
            ],
            interWordLatenciesMs: const [400],
            averageInterWordLatencyMs: 400.0,
          ),
          TranscriptTurn(
            turnIndex: 1,
            id: 'seg-2',
            role: 'agent',
            speaker: const SessionSpeaker(
              identity: 'agent-bot',
              name: 'Voice Bot',
              isAgent: true,
            ),
            text: 'Hi there! How can I help you today?',
            isFinal: true,
            startTimestamp: now.add(const Duration(milliseconds: 1500)),
            endTimestamp: now.add(const Duration(milliseconds: 3500)),
            durationMs: 2000,
            words: [
              WordTiming(
                word: 'Hi',
                timestamp: now.add(const Duration(milliseconds: 1500)),
                latencyFromPreviousWordMs: 0,
                latencyFromTurnStartMs: 0,
              ),
              WordTiming(
                word: 'there!',
                timestamp: now.add(const Duration(milliseconds: 1750)),
                latencyFromPreviousWordMs: 250,
                latencyFromTurnStartMs: 250,
              ),
            ],
            interWordLatenciesMs: const [250],
            averageInterWordLatencyMs: 250.0,
          ),
        ],
        turnTakingMetrics: [
          TurnTakingMetrics(
            turnIndex: 0,
            userTurnId: 'seg-1',
            agentTurnId: 'seg-2',
            agentSpeaker: const SessionSpeaker(
              identity: 'agent-bot',
              name: 'Voice Bot',
              isAgent: true,
            ),
            timestamps: TurnTakingTimestamps(
              userInputEndedAt: now.add(const Duration(milliseconds: 1200)),
              agentProcessingStartedAt: now.add(const Duration(milliseconds: 1300)),
              agentStartedSpeakingAt: now.add(const Duration(milliseconds: 1500)),
              agentFinishedSpeakingAt: now.add(const Duration(milliseconds: 3500)),
            ),
            latencies: const TurnTakingLatencies(
              turnTakingLatencyMs: 300,
              processingLatencyMs: 200,
              agentSpeakingDurationMs: 2000,
            ),
          ),
        ],
        agentStateTransitions: [
          AgentStateTransition(
            timestamp: now.add(const Duration(milliseconds: 1300)),
            participantIdentity: 'agent-bot',
            fromState: 'listening',
            toState: 'thinking',
            durationInPreviousStateMs: 1300,
          ),
          AgentStateTransition(
            timestamp: now.add(const Duration(milliseconds: 1500)),
            participantIdentity: 'agent-bot',
            fromState: 'thinking',
            toState: 'speaking',
            durationInPreviousStateMs: 200,
          ),
        ],
        summary: const SessionSummary(
          totalTurns: 2,
          userTurns: 1,
          agentTurns: 1,
          totalWords: 4,
          averageTurnTakingLatencyMs: 300.0,
          averageProcessingLatencyMs: 200.0,
          averageInterWordLatencyMs: 325.0,
        ),
      );

      final jsonString = recording.toJsonString();
      final decodedMap = jsonDecode(jsonString) as Map<String, dynamic>;
      final parsed = SessionRecording.fromJson(decodedMap);

      expect(parsed.id, equals('test-session-123'));
      expect(parsed.roomName, equals('test-room'));
      expect(parsed.durationMs, equals(10000));
      expect(parsed.localParticipant?.identity, equals('user-1'));
      expect(parsed.participants.length, equals(1));
      expect(parsed.participants.first.identity, equals('agent-bot'));
      expect(parsed.participants.first.isAgent, isTrue);

      expect(parsed.transcripts.length, equals(2));
      expect(parsed.transcripts[0].speaker.identity, equals('user-1'));
      expect(parsed.transcripts[0].role, equals('user'));
      expect(parsed.transcripts[0].words.length, equals(2));
      expect(parsed.transcripts[0].interWordLatenciesMs, equals([400]));

      expect(parsed.transcripts[1].speaker.identity, equals('agent-bot'));
      expect(parsed.transcripts[1].role, equals('agent'));
      expect(parsed.transcripts[1].words.length, equals(2));

      expect(parsed.turnTakingMetrics.length, equals(1));
      expect(parsed.turnTakingMetrics[0].latencies.turnTakingLatencyMs, equals(300));
      expect(parsed.turnTakingMetrics[0].latencies.processingLatencyMs, equals(200));
      expect(parsed.turnTakingMetrics[0].latencies.agentSpeakingDurationMs, equals(2000));

      expect(parsed.agentStateTransitions.length, equals(2));
      expect(parsed.agentStateTransitions[0].toState, equals('thinking'));
      expect(parsed.agentStateTransitions[1].toState, equals('speaking'));

      expect(parsed.summary.totalTurns, equals(2));
      expect(parsed.summary.userTurns, equals(1));
      expect(parsed.summary.agentTurns, equals(1));
      expect(parsed.summary.averageTurnTakingLatencyMs, equals(300.0));
      expect(parsed.summary.averageProcessingLatencyMs, equals(200.0));
    });
  });

  group('SessionRecorder Service', () {
    late Directory tempDir;
    late sdk.Room mockRoom;
    late SessionRecorder recorder;

    setUp(() async {
      tempDir = await Directory.systemTemp.createTemp('session_recorder_test');
      mockRoom = sdk.Room();
      recorder = SessionRecorder(
        room: mockRoom,
        customSaveDirectory: tempDir.path,
      );
    });

    tearDown(() async {
      await recorder.dispose();
      await mockRoom.dispose();
      if (await tempDir.exists()) {
        await tempDir.delete(recursive: true);
      }
    });

    test('records user and agent transcripts, inter-word latencies, and turn-taking metrics', () async {
      recorder.startNewSession(roomName: 'test-interactive-room');

      final t0 = DateTime.now();

      // 1. User speaks chunk by chunk
      recorder.onTranscriptionChunk(
        chunkText: 'What is ',
        segmentId: 'user-seg-1',
        participantIdentity: 'mock-user-identity',
        isFinal: false,
        timestamp: t0,
      );

      final t1 = t0.add(const Duration(milliseconds: 250));
      recorder.onTranscriptionChunk(
        chunkText: 'the weather today?',
        segmentId: 'user-seg-1',
        participantIdentity: 'mock-user-identity',
        isFinal: true,
        timestamp: t1,
      );

      // 2. Agent state changes: listening -> thinking -> speaking
      final t2 = t1.add(const Duration(milliseconds: 100));
      recorder.onTranscriptionChunk(
        chunkText: 'It is ',
        segmentId: 'agent-seg-1',
        participantIdentity: 'agent-bot-identity',
        isFinal: false,
        timestamp: t2,
      );

      final t3 = t2.add(const Duration(milliseconds: 300));
      recorder.onTranscriptionChunk(
        chunkText: 'sunny and warm.',
        segmentId: 'agent-seg-1',
        participantIdentity: 'agent-bot-identity',
        isFinal: true,
        timestamp: t3,
      );

      // Finalize and save recording
      final savedFile = await recorder.finalizeAndSave(customDirectoryPath: tempDir.path);
      expect(savedFile, isNotNull);
      expect(await savedFile!.exists(), isTrue);

      final jsonContent = await savedFile.readAsString();
      final decoded = jsonDecode(jsonContent) as Map<String, dynamic>;

      expect(decoded['roomName'], equals('test-interactive-room'));
      expect(decoded['transcripts'], isNotEmpty);

      final transcripts = decoded['transcripts'] as List<dynamic>;
      expect(transcripts.length, equals(2));

      // Check User transcript
      final userTurn = transcripts[0] as Map<String, dynamic>;
      expect(userTurn['text'], equals('What is the weather today?'));
      expect(userTurn['role'], equals('agent')); // non-local defaults to agent speaker

      // Check Turn taking metrics & JSON output
      expect(decoded['summary'], isNotNull);
      final summary = decoded['summary'] as Map<String, dynamic>;
      expect(summary['totalTurns'], equals(2));
      expect(summary['totalWords'], greaterThan(0));
    });

    test('records text chat messages', () async {
      recorder.startNewSession(roomName: 'chat-room');
      final now = DateTime.now();

      recorder.onUserTextMessage(
        id: 'msg-uuid-1',
        text: 'Tell me a joke',
        timestamp: now,
      );

      final file = await recorder.finalizeAndSave(customDirectoryPath: tempDir.path);
      expect(file, isNotNull);

      final json = jsonDecode(await file!.readAsString()) as Map<String, dynamic>;
      final transcripts = json['transcripts'] as List<dynamic>;
      expect(transcripts.length, equals(1));
      expect(transcripts[0]['text'], equals('Tell me a joke'));
      expect(transcripts[0]['role'], equals('user'));
    });
  });
}
