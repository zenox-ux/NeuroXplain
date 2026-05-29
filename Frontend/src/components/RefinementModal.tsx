import React, { useState, useRef } from 'react';
import * as d3 from 'd3';
import { X, Trash2, MoveHorizontal } from 'lucide-react';
import { AnalysisResult, ParsedEDF, SignalHeader } from '../types';

const ABNORMALITY_LABELS = ["Seizure", "Spike", "Sharp Wave", "Slow Activity", "Artifact", "Other"];

interface RefinementModalProps {
  boxes: AnalysisResult[];
  data: ParsedEDF;
  displaySignals: SignalHeader[];
  amplitudeScale: number;
  onConfirm: (label: string, refinedStart: number, refinedEnd: number) => void;
  onDelete: (ids: string[]) => void;
  onCancel: () => void;
}

const RefinementModal: React.FC<RefinementModalProps> = ({ 
  boxes, data, displaySignals, amplitudeScale, onConfirm, onDelete, onCancel 
}) => {
  const [refinedStart, setRefinedStart] = useState(boxes[0].global_start_time_sec);
  const [refinedEnd, setRefinedEnd] = useState(boxes[0].global_end_time_sec);
  const [isDragging, setIsDragging] = useState<'start' | 'end' | null>(null);

  const modalRef = useRef<SVGSVGElement>(null);
  const buffer = 2.0; 
  const width = 1000;
  const laneHeight = 150; 
  const viewStart = Math.max(0, boxes[0].global_start_time_sec - buffer);
  const viewEnd = Math.min(data.totalDuration, boxes[0].global_end_time_sec + buffer);
  const totalViewDuration = viewEnd - viewStart;

  const getPathForChannel = (channelName: string, h: number) => {
    const sig = displaySignals.find(s => s.label === channelName);
    if (!sig) return "";
    const rawData = data.data[data.signals.findIndex(s => s.label === sig.label)];
    const sr = sig.samplesPerRecord / data.header.recordDuration;
    const startIdx = Math.floor(viewStart * sr);
    const endIdx = Math.ceil(viewEnd * sr);
    const slice = rawData.subarray(startIdx, endIdx);
    const x = d3.scaleLinear().domain([0, slice.length]).range([0, width]);
    const y = d3.scaleLinear().domain([-100/amplitudeScale, 100/amplitudeScale]).range([h, 0]);
    return d3.line<number>().x((_, i) => x(i)).y(d => y(d))(Array.from(slice)) || "";
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isDragging || !modalRef.current) return;
    const rect = modalRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    let timeAtMouse = viewStart + (x / width) * totalViewDuration;
    timeAtMouse = Math.max(viewStart, Math.min(viewEnd, timeAtMouse));
    if (isDragging === 'start') setRefinedStart(Math.min(timeAtMouse, refinedEnd - 0.05));
    else setRefinedEnd(Math.max(timeAtMouse, refinedStart + 0.05));
  };

  const startX = ((refinedStart - viewStart) / totalViewDuration) * width;
  const endX = ((refinedEnd - viewStart) / totalViewDuration) * width;

  return (
    <div className="fixed inset-0 z-[200] flex items-center justify-center bg-slate-950/80 backdrop-blur-md p-6" onMouseMove={handleMouseMove} onMouseUp={() => setIsDragging(null)}>
      <div className="bg-slate-900 border border-slate-700 rounded-3xl w-full max-w-6xl shadow-2xl flex flex-col overflow-hidden max-h-[90vh]">
        <div className="p-5 border-b border-slate-800 flex justify-between items-center bg-slate-800/30">
          <div>
            <h3 className="text-lg font-bold text-white flex items-center">
              <MoveHorizontal className="mr-2 text-indigo-400" size={20} /> 
              Refining {boxes.length} Channels
            </h3>
          </div>
          <button onClick={onCancel} className="p-2 hover:bg-slate-700 text-slate-400 rounded-full transition-all"><X size={20} /></button>
        </div>

        <div className="flex-1 overflow-y-auto bg-slate-950 p-8 flex flex-col items-center">
          <svg ref={modalRef} width={width} height={boxes.length * laneHeight} className="overflow-visible select-none">
            {boxes.map((box, i) => (
              <g key={box.channel_name} transform={`translate(0, ${i * laneHeight})`}>
                <text x="-10" y={laneHeight / 2} textAnchor="end" className="fill-slate-500 text-[10px] font-bold">{box.channel_name}</text>
                <line x1="0" y1={laneHeight} x2={width} y2={laneHeight} stroke="#1e293b" strokeWidth="1" />
                <path d={getPathForChannel(box.channel_name, laneHeight)} fill="none" stroke="#38bdf8" strokeWidth="1.5" className="opacity-80" />
                <rect x={startX} y="0" width={endX - startX} height={laneHeight} fill="rgba(99, 102, 241, 0.1)" />
              </g>
            ))}

            <g className="cursor-ew-resize group" onMouseDown={() => setIsDragging('start')}>
              <rect x={startX - 20} y="0" width="40" height={boxes.length * laneHeight} fill="transparent" className="pointer-events-auto" />
              <line x1={startX} x2={startX} y1="0" y2={boxes.length * laneHeight} stroke="#ef4444" strokeWidth="2" className="group-hover:stroke-red-400 group-hover:stroke-[3px] transition-all" />
              <circle cx={startX} cy="0" r="6" fill="#ef4444" />
            </g>
            <g className="cursor-ew-resize group" onMouseDown={() => setIsDragging('end')}>
              <rect x={endX - 20} y="0" width="40" height={boxes.length * laneHeight} fill="transparent" className="pointer-events-auto" />
              <line x1={endX} x2={endX} y1="0" y2={boxes.length * laneHeight} stroke="#ef4444" strokeWidth="2" className="group-hover:stroke-red-400 group-hover:stroke-[3px] transition-all" />
              <circle cx={endX} cy={boxes.length * laneHeight} r="6" fill="#ef4444" />
            </g>
          </svg>
        </div>

        <div className="p-6 border-t border-slate-800 bg-slate-900 flex justify-between items-center">
          <div className="flex flex-wrap gap-2">
            {ABNORMALITY_LABELS.map(lab => (
              <button key={lab} onClick={() => onConfirm(lab, refinedStart, refinedEnd)} className="px-4 py-2 bg-slate-800 hover:bg-indigo-600 border border-slate-700 rounded-lg text-xs font-bold text-slate-200 transition-all">{lab}</button>
            ))}
          </div>
          <button onClick={() => onDelete(boxes.map(b => b.window_id))} className="flex items-center space-x-2 px-4 py-2 bg-red-900/20 hover:bg-red-600 text-red-400 hover:text-white rounded-lg text-xs font-bold transition-all">
            <Trash2 size={16} /> <span>Delete All</span>
          </button>
        </div>
      </div>
    </div>
  );
};

export default RefinementModal;