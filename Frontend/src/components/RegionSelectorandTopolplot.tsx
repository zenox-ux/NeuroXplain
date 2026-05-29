import React from 'react';
import { Layers } from 'lucide-react';
import { RegionSelectorandTopolplotProps, AnalysisResult } from '../types';

interface ExtendedProps extends RegionSelectorandTopolplotProps {
  onExplainRequest: (result: AnalysisResult) => void;
}

const RegionSelectorandTopolplot: React.FC<ExtendedProps> = ({
  analysisResults, selectedResult, setSelectedResult, currentTime, setCurrentTime, isPlaying, setIsPlaying, formatTime, onExplainRequest
}) => {
  return (
    <div className="w-80 flex bg-slate-900">
      <div className="flex-1 flex flex-col border-r border-slate-800">
        <div className="p-3 border-b border-slate-800 bg-slate-900/50 flex justify-between items-center font-bold text-xs text-slate-400 uppercase tracking-wider">
          <span><Layers className="inline w-3 h-3 mr-2" /> Detections</span>
          <span className="bg-slate-800 px-2 py-0.5 rounded-full">{analysisResults.length}</span>
        </div>
        <div className="flex-1 overflow-y-auto p-2 space-y-2">
          {analysisResults.map((res, i) => (
            <div
              key={res.window_id}
              className={`w-full p-3 rounded-lg border transition-all ${selectedResult === res ? "bg-indigo-900/20 border-indigo-500" : "bg-slate-800/40 border-slate-700"}`}
            >
              {/* Top row: navigate to segment */}
              <button
                onClick={() => {
                  setSelectedResult(res);
                  setCurrentTime(Math.max(0, res.global_start_time_sec - 1));
                  setIsPlaying(false);
                }}
                className="w-full text-left mb-2"
              >
                <div className="flex justify-between mb-1">
                  <span className="text-xs font-bold text-slate-300">
                    {res.label || `Event #${i + 1}`}
                  </span>
                  <span className="text-[10px] bg-slate-700 px-1.5 rounded">{res.channel_name}</span>
                </div>
                <div className="text-[10px] text-slate-500 font-mono">
                  {formatTime(res.global_start_time_sec)} - {formatTime(res.global_end_time_sec)}
                </div>
              </button>

              {/* Bottom row: Explain button — THIS triggers the topoplot */}
              <button
                onClick={() => {
                  setSelectedResult(res);
                  onExplainRequest(res);
                }}
                className="w-full mt-1 py-1 px-2 bg-indigo-700 hover:bg-indigo-600 text-white text-[10px] font-bold rounded transition-all"
              >
                 Explain
              </button>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default RegionSelectorandTopolplot;