import React, { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import RefinementModal from './RefinementModal';
import { generateEEGReport } from '../utils/ReportGenerator';
import InterpretationSuite from './InterpretationSuite';
import { InterpretationResponse } from '../types';
import * as d3 from 'd3';
import { 
  Play, 
  Pause, 
  Activity, 
  SkipBack, 
  SkipForward, 
  ZoomIn, 
  ZoomOut, 
  X, 
  Clock, 
  Save,
  AlertCircle,
  ChevronRight,
  FileDown
} from 'lucide-react';
import { ParsedEDF, AnalysisResult, CSVMetadata } from '../types';
import RegionSelectorandTopolplot from './RegionSelectorandTopolplot';

interface EEGViewerProps { 
  data: ParsedEDF;
  fileName: string;
  analysisResults: AnalysisResult[]; 
  csvMetadata: CSVMetadata | null;
  selectedResult: AnalysisResult | null;
  setSelectedResult: React.Dispatch<React.SetStateAction<AnalysisResult | null>>;
  setCurrentTime: React.Dispatch<React.SetStateAction<number>>;
  currentTime: number;
  isPlaying: boolean;
  setIsPlaying: React.Dispatch<React.SetStateAction<boolean>>;
  onReset: () => void;
}

const EEGViewer: React.FC<EEGViewerProps> = ({ 
  data, fileName, analysisResults, csvMetadata, selectedResult, setSelectedResult, 
  currentTime, setCurrentTime, isPlaying, setIsPlaying, onReset 
}) => {
  // --- UI Configuration ---
  const [windowSize, setWindowSize] = useState(30); 
  const [amplitudeScale, setAmplitudeScale] = useState(25);
  const [localEvents, setLocalEvents] = useState<AnalysisResult[]>([]);
  
  // --- Interaction State ---
  const [drawStart, setDrawStart] = useState<{ x: number, t: number, channelIdx: number } | null>(null);
  const [tempBox, setTempBox] = useState<{ x: number, w: number, startIdx: number, endIdx: number } | null>(null);
  const [refineBoxes, setRefineBoxes] = useState<AnalysisResult[] | null>(null);
  const [isGeneratingReport, setIsGeneratingReport] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);
  const animationRef = useRef<number | null>(null);
  const lastTimeRef = useRef<number>(0);
  const [interpretation, setInterpretation] = useState<InterpretationResponse | null>(null);
  const [isAIProcessing, setIsAIProcessing] = useState(false);

  const GRID_COLOR = "#1e293b"; 
  const SIGNAL_COLOR = "#38bdf8"; 
  const BOX_COLOR = "rgba(99, 102, 241, 0.2)";
  const BOX_STROKE = "#818cf8";

  useEffect(() => {
    setLocalEvents(analysisResults);
  }, [analysisResults]);

  const displaySignals = useMemo(() => 
    data.signals.filter(s => s.label.toLowerCase() !== 'edf annotations'), 
  [data]);
  
  const numChannels = displaySignals.length;

  // --- Playback Engine ---
  const animate = useCallback((timestamp: number) => {
    if (!lastTimeRef.current) lastTimeRef.current = timestamp;
    const deltaTime = (timestamp - lastTimeRef.current) / 1000;
    
    setCurrentTime(prev => {
      let next = prev + deltaTime;
      if (next > data.totalDuration - windowSize) { 
        setIsPlaying(false); 
        return prev; 
      }
      return next;
    });

    lastTimeRef.current = timestamp;
    if (isPlaying) {
        animationRef.current = requestAnimationFrame(animate);
    }
  }, [isPlaying, data.totalDuration, windowSize, setCurrentTime, setIsPlaying]);

  useEffect(() => {
    if (isPlaying) { 
      lastTimeRef.current = 0; 
      animationRef.current = requestAnimationFrame(animate); 
    } else if (animationRef.current) {
      cancelAnimationFrame(animationRef.current);
    }
    return () => { if (animationRef.current) cancelAnimationFrame(animationRef.current); };
  }, [isPlaying, animate]);

  // --- Interaction Handlers ---
  const handleMouseDownLane = (e: React.MouseEvent, index: number) => {
    if (refineBoxes) return;
    const rect = containerRef.current?.getBoundingClientRect();
    if (!rect) return;
    
    const x = e.clientX - rect.left;
    setDrawStart({ 
      x, 
      t: currentTime + (x / rect.width) * windowSize, 
      channelIdx: index 
    });
  };

  const handleMouseDownBox = (e: React.MouseEvent, box: AnalysisResult) => {
    e.stopPropagation();
    setSelectedResult(box);
    setRefineBoxes([box]); 
  };
  const handleDownloadReport = async () => {
  setIsGeneratingReport(true);
  try {
    await generateEEGReport(fileName, data, localEvents, formatTime);
  } catch (err) {
    alert("Failed to generate report. Please check Gemini API connection.");
  } finally {
    setIsGeneratingReport(false);
  }
};

  const handleMouseMove = (e: React.MouseEvent) => {
    const rect = containerRef.current?.getBoundingClientRect();
    if (!rect || !drawStart) return;

    const currentX = e.clientX - rect.left;
    const currentY = e.clientY - rect.top;
    const h = rect.height / numChannels;
    const currentChannelIdx = Math.max(0, Math.min(numChannels - 1, Math.floor(currentY / h)));

    setTempBox({ 
      x: Math.min(drawStart.x, currentX), 
      w: Math.abs(currentX - drawStart.x), 
      startIdx: Math.min(drawStart.channelIdx, currentChannelIdx),
      endIdx: Math.max(drawStart.channelIdx, currentChannelIdx)
    });
  };

  const handleMouseUp = (e: React.MouseEvent) => {
    if (drawStart && tempBox && tempBox.w > 5) {
      const rect = containerRef.current?.getBoundingClientRect();
      if (rect) {
        const x = Math.max(0, Math.min(e.clientX - rect.left, rect.width));
        const timeAtEnd = currentTime + (x / rect.width) * windowSize;
        const startTime = Math.max(0, Math.min(drawStart.t, timeAtEnd));
        const endTime = Math.min(data.totalDuration, Math.max(drawStart.t, timeAtEnd));

        const newEvents: AnalysisResult[] = [];
        for (let i = tempBox.startIdx; i <= tempBox.endIdx; i++) {
          newEvents.push({
            window_id: `evt_${Date.now()}_${i}`, 
            classification: 'abnormal',
            channel_name: displaySignals[i].label, 
            global_start_time_sec: startTime,
            global_end_time_sec: endTime,
            label: "Unspecified",
            event: [{ confidence: 1.0, region_start_time_sec: startTime, region_end_time_sec: endTime }]
          });
        }
        setLocalEvents(prev => [...prev, ...newEvents]);
        setRefineBoxes(newEvents); 
      }
    }
    setDrawStart(null);
    setTempBox(null);
  };

  const handleSave = () => {
    const headers = "Gender,Age,File Start,Start time,End time,Channel names,Comment\n";
    const toAbsoluteTime = (relativeSec: number) => {
      const baseTime = csvMetadata?.fileStart || "00:00:00";
      const parts = baseTime.split(':');
      const baseTotalSec = (parseInt(parts[0]) * 3600) + (parseInt(parts[1]) * 60) + parseInt(parts[2]);
      const currentTotalSec = baseTotalSec + relativeSec;
      const h = Math.floor(currentTotalSec / 3600).toString().padStart(2, '0');
      const m = Math.floor((currentTotalSec % 3600) / 60).toString().padStart(2, '0');
      const s = Math.floor(currentTotalSec % 60).toString().padStart(2, '0');
      const ms = Math.floor((currentTotalSec % 1) * 1000).toString().padStart(3, '0');
      return `${h}:${m}:${s}:${ms}`;
    };

    const rows = localEvents.map((ev, index) => {
      const startStr = toAbsoluteTime(ev.global_start_time_sec);
      const endStr = toAbsoluteTime(ev.global_end_time_sec);
      return index === 0 
        ? `${csvMetadata?.gender || ''},${csvMetadata?.age || ''},${csvMetadata?.fileStart || ''},${startStr},${endStr},${ev.channel_name},${ev.label || 'Abnormal'}`
        : `,,,${startStr},${endStr},${ev.channel_name},${ev.label || 'Abnormal'}`;
    }).join('\n');

    const blob = new Blob([headers + rows], { type: 'text/csv' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = `refined_${fileName.split('.')[0]}.csv`;
    link.click();
  };

  const getChannelPath = (idx: number, w: number, h: number) => {
    const sig = displaySignals[idx];
    const sigIndex = data.signals.indexOf(sig);
    const rawData = data.data[sigIndex];
    if (!rawData) return "";

    const sr = sig.samplesPerRecord / data.header.recordDuration;
    const slice = rawData.subarray(
        Math.max(0, Math.floor(currentTime * sr)), 
        Math.min(rawData.length, Math.ceil((currentTime + windowSize) * sr))
    );
    if (slice.length === 0) return "";
    const x = d3.scaleLinear().domain([0, slice.length]).range([0, w]);
    const y = d3.scaleLinear().domain([-100/amplitudeScale, 100/amplitudeScale]).range([h/2, -h/2]);
    return d3.line<number>().x((_, i) => x(i)).y(d => y(d))(Array.from(slice)) || "";
  };

  const formatTime = (seconds: number) => {
    const h = Math.floor(seconds / 3600).toString().padStart(2, '0');
    const m = Math.floor((seconds % 3600) / 60).toString().padStart(2, '0');
    const s = Math.floor(seconds % 60).toString().padStart(2, '0');
    return `${h}:${m}:${s}`;
  };

  const fetchExplanation = async (box: AnalysisResult) => {
  setIsAIProcessing(true);
  setInterpretation(null);
  try {
    const samplingRate = 200;
    const startIdx = Math.floor(box.global_start_time_sec * samplingRate);
    const signalMatrix = data.data.map(channelArray =>
      Array.from(channelArray.subarray(startIdx, startIdx + 128))
    );
    const response = await fetch("https://5c7a-2407-d000-17-3b14-29d6-56e5-21f6-8c12.ngrok-free.app/explain", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ signal: signalMatrix })
    });
    if (!response.ok) throw new Error("AI Backend failed");
    const result = await response.json();
    setInterpretation(result);
  } catch (err) {
    console.error("AI interpretation failed:", err);
    alert("Could not reach AI Backend.");
  } finally {
    setIsAIProcessing(false);
  }
};

  return (
<div className="flex flex-col bg-slate-950 text-slate-200 font-sans overflow-y-auto">
      
      {/* 1. TOP HEADER */}
      <div className="h-screen flex flex-col overflow-hidden flex-shrink-0">
      <header className="h-16 flex items-center justify-between px-6 bg-slate-900 border-b border-slate-800 z-50">
        <div className="flex items-center space-x-6">
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-indigo-600 rounded-lg shadow-lg"><Activity size={20} className="text-white" /></div>
            <div>
              <h1 className="text-sm font-bold text-white tracking-wide uppercase">{fileName}</h1>
              <div className="text-[10px] text-slate-500 font-mono">ID: {data.header.patientId || "ANONYMOUS"} | {new Date(data.header.startDate).toLocaleDateString()}</div>
            </div>
          </div>
          <button onClick={handleSave} className="flex items-center space-x-2 px-4 py-2 bg-emerald-600 hover:bg-emerald-500 text-white rounded-md text-xs font-bold transition shadow-lg"><Save size={14} /> <span>Export CSV</span></button>
        </div>
                <button 
          onClick={handleDownloadReport}
          disabled={isGeneratingReport}
          className={`flex items-center space-x-2 px-4 py-2 rounded-lg text-xs font-bold transition-all shadow-lg 
            ${isGeneratingReport ? 'bg-slate-700 cursor-wait' : 'bg-indigo-600 hover:bg-indigo-500 text-white'}`}
        >
          {isGeneratingReport ? (
            <div className="flex items-center space-x-2">
              <div className="w-3 h-3 border-2 border-white/30 border-t-white rounded-full animate-spin"></div>
              <span>ANALYZING...</span>
            </div>
          ) : (
            <>
              <FileDown size={14} /> 
              <span>GENERATE PDF REPORT</span>
            </>
          )}
        </button>
        <div className="flex items-center space-x-6">
          <div className="flex items-center bg-slate-800 rounded-lg p-1 border border-slate-700">
            <button onClick={() => setCurrentTime(prev => Math.max(0, prev - 5))} className="p-2 hover:bg-slate-700 rounded-md transition text-slate-400"><SkipBack size={16} /></button>
            <button onClick={() => setIsPlaying(!isPlaying)} className={`p-2 mx-1 rounded-md transition ${isPlaying ? 'bg-indigo-500 text-white shadow-lg' : 'bg-slate-700 text-slate-200 hover:bg-slate-600'}`}>
              {isPlaying ? <Pause size={16} fill="currentColor" /> : <Play size={16} fill="currentColor" className="ml-0.5" />}
            </button>
            <button onClick={() => setCurrentTime(prev => Math.min(data.totalDuration, prev + 5))} className="p-2 hover:bg-slate-700 rounded-md transition text-slate-400"><SkipForward size={16} /></button>
          </div>
          <div className="flex items-center space-x-3">
             <div className="flex items-center bg-slate-800 rounded-lg border border-slate-700 px-2 h-10 space-x-4">
                <div className="flex items-center space-x-2">
                  <ZoomIn size={12} className="text-slate-500" />
                  <span className="text-[10px] font-mono text-indigo-400 w-6 text-center">{windowSize}s</span>
                  <input type="range" min="1" max="60" value={windowSize} onChange={(e) => setWindowSize(Number(e.target.value))} className="w-16 h-1 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-indigo-500" />
                </div>
                <div className="w-px h-4 bg-slate-700" />
                <div className="flex items-center space-x-2">
                  <Activity size={12} className="text-slate-500" />
                  <span className="text-[10px] font-mono text-indigo-400 w-8 text-center">x{amplitudeScale}</span>
                  <input type="range" min="1" max="100" value={amplitudeScale} onChange={(e) => setAmplitudeScale(Number(e.target.value))} className="w-16 h-1 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-indigo-500" />
                </div>
             </div>
          </div>
          <button onClick={onReset} className="p-2 text-slate-500 hover:text-red-400 transition hover:bg-red-500/10 rounded-lg"><X size={20} /></button>
        </div>
      </header>

      {/* 2. MAIN CONTENT AREA (Signals + Detections Sidebar) */}
      <div className="flex-1 flex overflow-hidden">
        
        {/* Channel Labels (Left) */}
        <div className="w-20 bg-slate-900 border-r border-slate-800 flex flex-col">
          {displaySignals.map((sig, i) => (
            <div 
              key={i} 
              className={`flex items-center justify-end pr-3 text-[10px] font-bold transition-all border-b border-slate-800/30 ${selectedResult?.channel_name === sig.label ? 'text-indigo-400 bg-indigo-500/5' : 'text-slate-500'}`} 
              style={{ height: `${100 / numChannels}%` }}
            >
              {sig.label}
            </div>
          ))}
        </div>

        {/* Signal Canvas (Center) */}
        <div 
          className="flex-1 relative bg-[#070b14] cursor-crosshair overflow-hidden" 
          ref={containerRef} 
          onMouseMove={handleMouseMove} 
          onMouseUp={handleMouseUp} 
          onMouseLeave={() => { setDrawStart(null); setTempBox(null); }}
        >
          {containerRef.current && displaySignals.map((sig, i) => {
            const h = containerRef.current!.clientHeight / numChannels;
            const w = containerRef.current!.clientWidth;
            return (
              <div 
                key={i} 
                className="absolute w-full border-b border-slate-800/20" 
                style={{ height: h, top: i * h }} 
                onMouseDown={(e) => handleMouseDownLane(e, i)}
              >
                <svg className="w-full h-full overflow-visible pointer-events-none">
                  <g transform={`translate(0, ${h / 2})`}>
                    <path d={getChannelPath(i, w, h)} fill="none" stroke={SIGNAL_COLOR} strokeWidth="1" className="opacity-80" />
                    {localEvents.filter(ev => ev.channel_name === sig.label).map((box) => {
                      const startX = ((box.global_start_time_sec - currentTime) / windowSize) * w;
                      const widthX = ((box.global_end_time_sec - box.global_start_time_sec) / windowSize) * w;
                      return startX + widthX >= 0 && startX <= w && (
                        <rect 
                          key={box.window_id} 
                          x={startX} y={-h/2} width={widthX} height={h} 
                          className="pointer-events-auto cursor-pointer"
                          fill={selectedResult?.window_id === box.window_id ? "rgba(239, 68, 68, 0.4)" : BOX_COLOR} 
                          stroke={selectedResult?.window_id === box.window_id ? "#ef4444" : BOX_STROKE} 
                          onMouseDown={(e) => handleMouseDownBox(e, box)}
                        >
                           <title>{`Label: ${box.label}\nChannel: ${box.channel_name}`}</title>
                        </rect>
                      );
                    })}
                    {tempBox && i === 0 && (
                      <rect 
                        x={tempBox.x} 
                        y={(containerRef.current!.clientHeight / numChannels) * tempBox.startIdx - (h/2)} 
                        width={tempBox.w} 
                        height={(containerRef.current!.clientHeight / numChannels) * (tempBox.endIdx - tempBox.startIdx + 1)} 
                        fill="rgba(34, 197, 94, 0.3)" stroke="#22c55e" strokeDasharray="4" 
                      />
                    )}
                  </g>
                </svg>
              </div>
            );
          })}
        </div>

        {/* Detections Sidebar (Right) */}
        <div className="w-80 bg-slate-900 border-l border-slate-800 flex flex-col shadow-2xl z-40">
            <div className="p-4 border-b border-slate-800 flex justify-between items-center bg-slate-800/10">
                <span className="text-[10px] font-bold text-slate-500 uppercase tracking-widest flex items-center">
                    <AlertCircle size={14} className="mr-2 text-indigo-400" /> Total Events: {localEvents.length}
                </span>
            </div>
            <div className="flex-1 overflow-y-auto">
                <RegionSelectorandTopolplot 
                    analysisResults={localEvents} 
                    selectedResult={selectedResult} 
                    setSelectedResult={setSelectedResult} 
                    currentTime={currentTime} 
                    setCurrentTime={setCurrentTime} 
                    isPlaying={isPlaying} 
                    setIsPlaying={setIsPlaying} 
                    formatTime={formatTime} 
                    onExplainRequest={fetchExplanation}
                />
            </div>
        </div>
      </div>

      {/* 3. BOTTOM TIMELINE (Thin Scroll Bar) */}
      <div className="h-20 bg-slate-900 border-t border-slate-800 flex items-center px-6 space-x-6">
          <span className="text-[10px] font-mono text-slate-500 w-16">{formatTime(currentTime)}</span>
          <div className="flex-1 relative h-1.5 bg-slate-800 rounded-full group cursor-pointer">
              <div 
                className="absolute inset-y-0 bg-indigo-500/40 border-x border-indigo-400 group-hover:bg-indigo-500/60 transition-all" 
                style={{ left: `${(currentTime/data.totalDuration)*100}%`, width: `${(windowSize/data.totalDuration)*100}%` }} 
              />
              <input type="range" min={0} max={Math.max(0, data.totalDuration - windowSize)} step={0.1} value={currentTime} onChange={(e) => setCurrentTime(parseFloat(e.target.value))} className="absolute inset-0 w-full h-full opacity-0 cursor-pointer" />
          </div>
          <span className="text-[10px] font-mono text-slate-500 w-16 text-right">{formatTime(data.totalDuration)}</span>
      </div>

      {/* Refinement Modal Overlay */}
      {refineBoxes && (
        <RefinementModal 
          boxes={refineBoxes} data={data} displaySignals={displaySignals} amplitudeScale={amplitudeScale}
          onCancel={() => { setRefineBoxes(null); setSelectedResult(null); }}
          onDelete={(ids) => {
            setLocalEvents(prev => prev.filter(ev => !ids.includes(ev.window_id)));
            setRefineBoxes(null);
            setSelectedResult(null);
          }}
          onConfirm={(label, start, end) => {
            setLocalEvents(prev => {
              const targetIds = refineBoxes.map(b => b.window_id);
              return prev.map(ev => targetIds.includes(ev.window_id) ? { ...ev, label, global_start_time_sec: start, global_end_time_sec: end } : ev)
                .sort((a,b) => a.global_start_time_sec - b.global_start_time_sec);
            });
            setRefineBoxes(null);
            setSelectedResult(null);
          }}
        />
      )}
      </div>
{isAIProcessing && (
  <div className="fixed inset-0 flex items-center justify-center z-[300] bg-slate-950">
    <div className="text-center p-8">
      <div className="relative w-32 h-32 mx-auto mb-8">
        <div className="absolute inset-0 bg-indigo-500 rounded-full animate-ping opacity-10"></div>
        <div className="relative bg-slate-900 border border-indigo-500/30 rounded-3xl w-full h-full flex items-center justify-center shadow-2xl">
          <Activity className="w-12 h-12 text-indigo-400 animate-pulse" />
        </div>
      </div>
      <h2 className="text-2xl font-bold text-white mb-2 tracking-tight">NeuroXplain AI</h2>
      <p className="text-slate-400 text-sm font-mono max-w-xs mx-auto leading-relaxed">
        Searching Prototype Database...
      </p>
      <div className="mt-8 w-48 h-1 bg-slate-800 rounded-full mx-auto overflow-hidden">
        <div className="h-full bg-indigo-500 animate-[loading_2s_ease-in-out_infinite]"></div>
      </div>
    </div>
  </div>
)}

{interpretation && selectedResult && (
  <div className="relative z-40 bg-[#020617]">
    <InterpretationSuite key={selectedResult.window_id} data={interpretation} />
  </div>
)}
    </div>
  );
};

export default EEGViewer;