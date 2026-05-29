// src/utils/xaiService.ts
import { AnalysisResult } from '../types';

const BACKEND_URL = "https://2892-34-123-54-156.ngrok-free.app/analyze_eeg";

export const analyzeSegment = async (
  signal: number[], 
  channelName: string, 
  globalOffset: number, 
  sfreq: number = 200
): Promise<AnalysisResult[]> => {
  try {
    const response = await fetch(BACKEND_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "ngrok-skip-browser-warning": "69420", 
      },
      body: JSON.stringify({ signal, sfreq }),
    });

    if (!response.ok) return [];
    const data = await response.json();

    if (!data.is_abnormal) return [];

    // Map localized segments (0s-2s) to the global EEG timeline
    return data.events.map((event: any, idx: number) => ({
      window_id: `xai_${globalOffset}_${idx}`,
      classification: 'abnormal',
      channel_name: channelName,
      global_start_time_sec: globalOffset + event.start_time,
      global_end_time_sec: globalOffset + event.end_time,
      label: event.label, // e.g., "SPIKE_AND_WAVE" from prototype library
      event: [{ 
        confidence: event.confidence_score, 
        region_start_time_sec: event.start_time, 
        region_end_time_sec: event.end_time 
      }]
    }));
  } catch (error) {
    console.error("XAI Backend Error:", error);
    return [];
  }
};