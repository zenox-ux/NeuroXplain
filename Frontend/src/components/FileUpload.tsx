import React, { useState } from 'react';
import { Activity, FileText, CheckCircle2 } from 'lucide-react';

interface FileUploadProps {
  onFilesSelect: (edfFile: File) => void;
}

const FileUpload: React.FC<FileUploadProps> = ({ onFilesSelect }) => {
  const [dragActive, setDragActive] = useState(false);
  const [selectedEDF, setSelectedEDF] = useState<File | null>(null);

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") setDragActive(true);
    else if (e.type === "dragleave") setDragActive(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    const file = e.dataTransfer.files[0];
    if (file && file.name.toLowerCase().endsWith('.edf')) {
      setSelectedEDF(file);
    }
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen w-screen bg-slate-950 text-slate-200 p-4 relative overflow-hidden">
      
      {/* --- THE BACKGROUND GLOW (The colors that changed) --- */}
      <div className="absolute top-0 left-0 w-full h-full overflow-hidden pointer-events-none opacity-30">
        <div className="absolute top-[-10%] left-[-5%] w-[40%] h-[40%] bg-blue-600 rounded-full blur-[120px]"></div>
        <div className="absolute bottom-[-10%] right-[-5%] w-[40%] h-[40%] bg-indigo-600 rounded-full blur-[120px]"></div>
      </div>

      <div 
        className={`relative w-full max-w-2xl p-12 text-center border-2 border-dashed rounded-[2rem] transition-all duration-500 backdrop-blur-2xl z-10 
          ${dragActive 
            ? 'border-indigo-400 bg-slate-800/60 scale-[1.01] shadow-[0_0_50px_rgba(99,102,241,0.15)]' 
            : 'border-slate-800 bg-slate-900/40 hover:border-slate-700 shadow-2xl'}`}
        onDragEnter={handleDrag} onDragLeave={handleDrag} onDragOver={handleDrag} onDrop={handleDrop}
      >
        <div className="w-20 h-20 bg-gradient-to-br from-blue-500 to-indigo-600 rounded-2xl flex items-center justify-center mx-auto mb-8 shadow-lg transform rotate-3">
          <Activity className="w-10 h-10 text-white" />
        </div>
        
        <h2 className="text-4xl font-bold text-white mb-2 tracking-tight">NeuroXplain</h2>
        <p className="text-slate-400 mb-10 text-lg">WAVESAGE-Powered AI Analysis</p>
        
        <div className="mb-10">
          <label className="group relative flex flex-col items-center justify-center p-12 bg-slate-950/50 rounded-2xl border border-slate-800 hover:border-indigo-500/50 hover:bg-slate-900/80 transition-all cursor-pointer overflow-hidden">
            <div className="absolute inset-0 bg-indigo-500/5 opacity-0 group-hover:opacity-100 transition-opacity"></div>
            <FileText className={`w-12 h-12 mb-4 transition-colors ${selectedEDF ? 'text-emerald-400' : 'text-slate-500 group-hover:text-indigo-400'}`} />
            <span className="text-base font-semibold text-slate-300 mb-1 uppercase tracking-widest">EDF Source</span>
            <span className="text-xs text-slate-500 truncate max-w-[250px]">
              {selectedEDF ? selectedEDF.name : 'Drop file here or click to browse'}
            </span>
            <input type="file" accept=".edf" className="hidden" onChange={(e) => e.target.files?.[0] && setSelectedEDF(e.target.files[0])} />
            {selectedEDF && <CheckCircle2 className="absolute top-4 right-4 w-6 h-6 text-emerald-400" />}
          </label>
        </div>

        <button
          onClick={() => selectedEDF && onFilesSelect(selectedEDF)}
          disabled={!selectedEDF}
          className={`w-full py-4 rounded-xl font-bold text-lg tracking-wide transition-all duration-300 transform 
            ${selectedEDF 
              ? 'bg-indigo-600 hover:bg-indigo-500 text-white shadow-[0_0_20px_rgba(79,70,229,0.3)] hover:-translate-y-1' 
              : 'bg-slate-800 text-slate-600 cursor-not-allowed'}`}
        >
          Initialize XAI Analysis
        </button>
      </div>
    </div>
  );
};

export default FileUpload;