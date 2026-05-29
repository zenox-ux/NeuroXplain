import React, { useState } from 'react';
import FileUpload from './components/FileUpload';
import EEGViewer from './components/EEGViewer';
import { parseEDF } from './utils/parsers';
import { ParsedEDF, AnalysisResult, CSVMetadata } from './types';
import { Activity, AlertCircle } from 'lucide-react';

const App: React.FC = () => {
  // --- Data State ---
  const [parsedData, setParsedData] = useState<ParsedEDF | null>(null);
  const [analysisResults, setAnalysisResults] = useState<AnalysisResult[]>([]);
  const [csvMetadata, setCsvMetadata] = useState<CSVMetadata | null>(null);
  const [fileName, setFileName] = useState("");

  // --- UI/Playback State ---
  const [selectedResult, setSelectedResult] = useState<AnalysisResult | null>(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  
  // --- AI Backend State ---
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  /**
   * Orchestrates the WAVESAGE analysis pipeline.
   * 1. Parses EDF locally.
   * 2. Slices first 10 seconds.
   * 3. Calls Colab Backend for XAI localization and labeling.
   * 4. Transitions to Viewer.
   */
  const handleFilesSelect = async (edf: File) => {
  setIsAnalyzing(true);
  setError(null);

  try {
    // 1. Parse the Raw EDF signal locally
    const data = await parseEDF(edf);
    
    const sfreq = 200; // Frequency from your model config
    const secondsToAnalyze = 20;
    const sampleLimit = sfreq * secondsToAnalyze;

    // 2. Define non-brain channels to exclude (Reference, Ground, Artifacts)
    const blacklistedChannels = [
      'a1', 'a2','f3','c4' ,'t5','fz','cz','pz','t5','t6','p3','t4','o2','o1','f4','p4','ref', 'msg', 'edf annotations', 
      'body', 'ecg', 'status', 'trigger'
    ];

    // 3. Filter for valid cortical channels only
    const validSignals = data.signals.filter(s => {
      const label = s.label.toLowerCase();
      const isBlacklisted = blacklistedChannels.some(black => label.includes(black));
      const isEmpty = label.trim() === "";
      return !isBlacklisted && !isEmpty;
    });

    if (validSignals.length === 0) {
      throw new Error("No valid cortical channels found in this EDF file.");
    }

    // 4. Map and Slice data for the Backend (Matrix: Channels x Samples)
    const montageData = validSignals.map(sig => {
      const originalIdx = data.signals.indexOf(sig);
      const channelBuffer = data.data[originalIdx];
      
      // Ensure we don't slice beyond the file length
      const endSample = Math.min(sampleLimit, channelBuffer.length);
      return Array.from(channelBuffer.slice(0, endSample));
    });

    const channelNames = validSignals.map(s => s.label);

    console.log(`🚀 Sending ${validSignals.length} channels to WAVESAGE Backend...`);

    // 5. Send the full montage to the Ngrok Backend
    const BACKEND_URL = "https://2892-34-123-54-156.ngrok-free.app/analyze_eeg";
    
    const response = await fetch(BACKEND_URL, {
      method: "POST",
      headers: { 
        "Content-Type": "application/json", 
        "ngrok-skip-browser-warning": "69420" 
      },
      body: JSON.stringify({ 
        signals: montageData, 
        channel_names: channelNames,
        sfreq: sfreq 
      })
    });

    if (!response.ok) throw new Error(`Backend Error: ${response.status}`);
    
    const result = await response.json();
    
    if (result.status === "success") {
      // 6. Map Backend Clinical Events to Frontend AnalysisResult type
      const mappedResults: AnalysisResult[] = result.events.map((ev: any) => ({
        window_id: `xai_${ev.channel_name}_${ev.start_time}`,
        classification: 'abnormal',
        channel_name: ev.channel_name,
        global_start_time_sec: ev.start_time,
        global_end_time_sec: ev.end_time,
        label: ev.label, // Clinical Guess from KNN/Library
        event: [{ 
          confidence: ev.confidence, 
          region_start_time_sec: ev.start_time, 
          region_end_time_sec: ev.end_time 
        }]
      }));

      // 7. Update States in correct order to trigger Viewer
      setAnalysisResults(mappedResults);
      setParsedData(data);
      setFileName(edf.name);
      
      // Auto-generate metadata if CSV is missing
      setCsvMetadata({
        gender: data.header.patientId.toLowerCase().includes('f') ? 'Female' : 'Male',
        age: 'N/A',
        fileStart: data.header.startTime
      });
    } else {
      throw new Error("Backend failed to process analysis.");
    }

  } catch (err) {
    console.error("XAI Pipeline Failed:", err);
    setError(err instanceof Error ? err.message : "Connection to WAVESAGE failed.");
  } finally {
    setIsAnalyzing(false);
  }
};

  const handleReset = () => {
    setParsedData(null);
    setAnalysisResults([]);
    setCsvMetadata(null);
    setFileName("");
    setIsPlaying(false);
    setSelectedResult(null);
    setError(null);
  };

 return (
  /* The main container must always be full-screen */
  <main className="min-h-screen bg-slate-950 text-slate-200">
    
    {isAnalyzing ? (
      /* --- 1. LOADING SCREEN (Centered) --- */
      <div className="fixed inset-0 flex items-center justify-center z-[100] bg-slate-950">
        <div className="text-center p-8">
          <div className="relative w-32 h-32 mx-auto mb-8">
            <div className="absolute inset-0 bg-indigo-500 rounded-full animate-ping opacity-10"></div>
            <div className="relative bg-slate-900 border border-indigo-500/30 rounded-3xl w-full h-full flex items-center justify-center shadow-2xl">
              <Activity className="w-12 h-12 text-indigo-400 animate-pulse" />
            </div>
          </div>
          <h2 className="text-2xl font-bold text-white mb-2 tracking-tight">NeuroXplain AI</h2>
          <p className="text-slate-400 text-sm font-mono max-w-xs mx-auto leading-relaxed">
            Running Model...
          </p>
          <div className="mt-8 w-48 h-1 bg-slate-800 rounded-full mx-auto overflow-hidden">
            <div className="h-full bg-indigo-500 animate-[loading_2s_ease-in-out_infinite]"></div>
          </div>
        </div>
      </div>

    ) : error ? (
      /* --- 2. ERROR STATE (Centered) --- */
      <div className="fixed inset-0 flex items-center justify-center z-[100] bg-slate-950">
        <div className="text-center bg-slate-900 p-10 rounded-3xl border border-red-500/20 shadow-2xl">
          <AlertCircle className="w-16 h-16 text-red-500 mx-auto mb-4" />
          <h2 className="text-xl font-bold text-white mb-2">Connection Failed</h2>
          <p className="text-slate-400 text-sm mb-6 max-w-xs">{error}</p>
          <button 
            onClick={() => setError(null)}
            className="px-6 py-2 bg-slate-800 hover:bg-slate-700 text-white rounded-xl transition-all"
          >
            Try Again
          </button>
        </div>
      </div>

    ) : parsedData ? (
      /* --- 3. VIEWER MODE (Full Screen) --- */
      /* No centering here, let EEGViewer take the full width/height */
      <div className="w-full h-screen">
        <EEGViewer 
          data={parsedData} 
          fileName={fileName} 
          analysisResults={analysisResults}
          csvMetadata={csvMetadata}
          selectedResult={selectedResult} 
          setSelectedResult={setSelectedResult}
          currentTime={currentTime} 
          setCurrentTime={setCurrentTime}
          isPlaying={isPlaying} 
          setIsPlaying={setIsPlaying}
          onReset={handleReset}
        />
      </div>
    ) : (
      /* --- 4. INITIAL UPLOAD (Centered) --- */
      <div className="fixed inset-0 flex items-center justify-center bg-slate-950">
        <FileUpload onFilesSelect={handleFilesSelect} />
      </div>
    )}
  </main>
);
};

export default App;
