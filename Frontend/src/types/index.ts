export interface EDFHeader {
  version: string;
  patientId: string;
  recordId: string;
  startDate: string;
  startTime: string;
  headerBytes: number;
  reserved: string;
  numDataRecords: number;
  recordDuration: number;
  numSignals: number;
}

export interface SignalHeader {
  label: string;
  transducerType: string;
  physicalDimension: string;
  physicalMin: number;
  physicalMax: number;
  digitalMin: number;
  digitalMax: number;
  prefiltering: string;
  samplesPerRecord: number;
  offset?: number;
}


export interface InterpretationResponse {
  prediction: number;
  weights: number[];
  nodes: { label: string; x: number; y: number }[];
  links: [string, string][];
  neighbors: {
    id: string;
    signal: number[][];
  }[];
}

export interface ParsedEDF {
  header: EDFHeader;
  signals: SignalHeader[];
  data: Float32Array[];
  totalDuration: number;
}

export interface DetectionEvent {
  confidence: number;
  region_start_time_sec: number;
  region_end_time_sec: number;
}

export interface AnalysisResult {
  window_id: string;
  classification: string;
  channel_name: string;
  global_start_time_sec: number;
  global_end_time_sec: number;
  event: DetectionEvent[];
}

export interface RegionSelectorandTopolplotProps {
  analysisResults: AnalysisResult[];
  selectedResult: AnalysisResult | null;
  setSelectedResult: React.Dispatch<React.SetStateAction<AnalysisResult | null>>;
  currentTime: number;
  setCurrentTime: React.Dispatch<React.SetStateAction<number>>;
  isPlaying: boolean;
  setIsPlaying: React.Dispatch<React.SetStateAction<boolean>>;
  formatTime: (t: number) => string;
  onExplainRequest: (result: AnalysisResult) => void;
}

export interface AnalysisResult {
  window_id: string;
  classification: string;
  channel_name: string;
  global_start_time_sec: number;
  global_end_time_sec: number;
  event: DetectionEvent[];
  label?: string; // New field for specific abnormality names
}

// Add to src/types.ts
export interface RefinementModalProps {
  box: AnalysisResult;
  data: ParsedEDF;
  displaySignals: SignalHeader[];
  amplitudeScale: number;
  onConfirm: (label: string) => void;
  onCancel: () => void;
  formatTime: (s: number) => string;
}
export interface CSVMetadata {
  gender: string;
  age: string;
  fileStart: string;
}
export type MontageType = 'Referential' | 'Longitudinal Bipolar';