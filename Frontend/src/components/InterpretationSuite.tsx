import React from 'react';
import { InterpretationResponse } from '../types';

const CHANNEL_NAMES = ["Fp1-Avg", "F3-Avg", "C3-Avg", "P3-Avg", "F7-Avg", "T3-Avg", "T5-Avg", "O1-Avg", "Fz-Avg", "Cz-Avg", "Pz-Avg", "Fp2-Avg", "F4-Avg", "C4-Avg", "P4-Avg", "F8-Avg", "T4-Avg", "T6-Avg", "O2-Avg", "Fp1-F7", "F7-T3", "T3-T5", "T5-O1", "Fp2-F8", "F8-T4", "T4-T6", "T6-O2", "Fp1-F3", "F3-C3", "C3-P3", "P3-O1", "Fp2-F4", "F4-C4", "C4-P4", "P4-O2", "Fz-Cz", "Cz-Pz"];

interface Props {
  data: InterpretationResponse;
}

const InterpretationSuite: React.FC<Props> = ({ data }) => {
  const size = 280;
  const pad = 35;
  const MATCH_CARD_WIDTH = 150; // Increased width for spacious waves

  const getColor = (importance: number) => {
    const r = Math.round(173 - (173 - 0) * importance);
    const g = Math.round(216 - (216 - 0) * importance);
    const b = Math.round(230 - (230 - 139) * importance);
    return `rgb(${r}, ${g}, ${b})`;
  };

  const getX = (x: number) => pad + x * (size - pad * 2);
  const getY = (y: number) => size - (pad + y * (size - pad * 2));

  return (
    <div className="flex w-full bg-[#020617] border-t border-indigo-500/30 p-6 space-x-8 h-[600px] overflow-hidden animate-in slide-in-from-bottom duration-700">
      
      {/* SECTION 1: SPATIAL (Topoplot) - Fixed Width */}
      <div className="w-[300px] flex-shrink-0 flex flex-col items-center border-r border-slate-800 pr-6">
        <h3 className="text-indigo-400 font-black text-xs uppercase tracking-[0.2em] mb-6 text-center w-full">Spatial Evidence</h3>
        <div className="bg-white/5 p-4 rounded-3xl border border-white/5 relative">
          <svg width={size} height={size} className="overflow-visible">
            <circle cx={size/2} cy={size/2} r={size/2 - 5} fill="none" stroke="#334155" strokeWidth="2" />
            <path d={`M ${size/2-10} 5 L ${size/2} -10 L ${size/2+10} 5`} fill="none" stroke="#334155" strokeWidth="2" />
            {data.links.map((link, i) => {
              const s = data.nodes.find(n => n.label === link[0])!;
              const e = data.nodes.find(n => n.label === link[1])!;
              return (
                <line key={i} x1={getX(s.x)} y1={getY(s.y)} x2={getX(e.x)} y2={getY(e.y)} 
                      stroke={getColor(data.weights[i + 19])} strokeWidth="6" strokeLinecap="round" opacity="0.8" />
              );
            })}
            {data.nodes.map((n, i) => (
              <g key={i} transform={`translate(${getX(n.x)}, ${getY(n.y)})`}>
                <circle r={12} fill={getColor(data.weights[i])} />
                <text dy=".3em" textAnchor="middle" fontSize="7" fontWeight="bold" fill={data.weights[i] > 0.5 ? "white" : "#0f172a"}>{n.label}</text>
              </g>
            ))}
          </svg>
        </div>
        {/* <div className="mt-8 text-center">
            <p className="text-slate-500 text-[9px] uppercase font-black tracking-widest mb-1">Model Prediction</p>
            <div className="text-4xl font-black text-orange-500 drop-shadow-[0_0_15px_rgba(249,115,22,0.3)]">
                {data.prediction.toFixed(2)}%
            </div>
        </div> */}
      </div>

      {/* SECTION 2: IMPORTANCE BARS - Fixed Width */}
      <div className="w-[350px] flex-shrink-0 flex flex-col border-r border-slate-800 pr-6">
        <h3 className="text-indigo-400 font-black text-xs uppercase tracking-[0.2em] mb-6">Channel Significance</h3>
        <div className="flex-1 overflow-y-auto space-y-1 pr-2 custom-scrollbar">
          {CHANNEL_NAMES.map((name, i) => (
            <div key={i} className="flex items-center h-4 group">
              <div className="w-16 text-[8px] font-mono text-slate-500 group-hover:text-indigo-300 transition-colors">{name}</div>
              <div className="flex-1 bg-slate-800/50 h-2 rounded-full overflow-hidden border border-white/5">
                <div 
                  className="h-full bg-gradient-to-r from-blue-600 to-indigo-400 transition-all duration-1000 ease-out"
                  style={{ width: `${data.weights[i] * 100}%` }} 
                />
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* SECTION 3: PROTOTYPE MATCHES - Spaced Out with Horizontal Scroll */}
      <div className="flex-1 flex flex-col min-w-0">
        <h3 className="text-emerald-400 font-black text-xs uppercase tracking-[0.2em] mb-6">Historical Evidence (Best Matches)</h3>
        
        {/* HORIZONTAL SCROLL CONTAINER */}
        <div className="flex-1 flex space-x-6 overflow-x-auto pb-4 custom-scrollbar-horizontal">
          {data.neighbors.map((neighbor, nIdx) => (
            <div 
              key={nIdx} 
              className="flex-shrink-0 flex flex-col bg-slate-900/40 border border-white/5 rounded-2xl p-4 shadow-2xl"
              style={{ width: `${MATCH_CARD_WIDTH}px` }}
            >
               <div className="flex justify-between items-center mb-4 border-b border-white/5 pb-2">
                  <span className="text-[10px] font-black text-emerald-400 uppercase tracking-widest">#{nIdx + 1} Best Match</span>
                  <span className="text-[8px] font-mono text-slate-500 truncate max-w-[200px]">{neighbor.id}</span>
               </div>

               {/* WAVEFORM AREA */}
               <div className="flex-1 flex flex-col space-y-1 overflow-hidden">
                  {neighbor.signal.map((samples, chIdx) => {
                    // Amplitude Scaling Logic
                    const min = Math.min(...samples);
                    const max = Math.max(...samples);
                    const range = max - min || 1;
                    
                    // X-axis now stretches across the full MATCH_CARD_WIDTH
                    // Y-axis is centered in a 16px high lane
                    const points = samples.map((v, sIdx) => {
                        const px = (sIdx / (samples.length - 1)) * (MATCH_CARD_WIDTH - 32); 
                        const py = 8 - ((v - min) / range) * 16;
                        return `${px},${py}`;
                    }).join(' ');

                    return (
                      <div key={chIdx} className="h-4 border-b border-white/5 flex items-center relative group/lane">
                          <svg width="100%" height="16" className="overflow-visible opacity-60 group-hover/lane:opacity-100 transition-opacity">
                              <polyline 
                                points={points} 
                                fill="none" 
                                stroke="#94a3b8" 
                                strokeWidth="1" 
                                strokeLinejoin="round"
                              />
                          </svg>
                      </div>
                    )
                  })}
               </div>
            </div>
          ))}
        </div>
      </div>

      {/* Global CSS for Custom Scrollbars */}
      <style>{`
        .custom-scrollbar::-webkit-scrollbar { width: 4px; }
        .custom-scrollbar::-webkit-scrollbar-thumb { background: #334155; border-radius: 10px; }
        
        .custom-scrollbar-horizontal::-webkit-scrollbar { height: 6px; }
        .custom-scrollbar-horizontal::-webkit-scrollbar-track { background: #0f172a; border-radius: 10px; }
        .custom-scrollbar-horizontal::-webkit-scrollbar-thumb { background: #312e81; border-radius: 10px; }
        .custom-scrollbar-horizontal::-webkit-scrollbar-thumb:hover { background: #4338ca; }
      `}</style>
    </div>
  );
};

export default InterpretationSuite;