import { EDFHeader, SignalHeader, ParsedEDF, AnalysisResult,CSVMetadata } from '../types';

const parseString = (buffer: ArrayBuffer, start: number, length: number): string => {
  const decoder = new TextDecoder('ascii');
  return decoder.decode(buffer.slice(start, start + length)).trim();
};

const parseNumber = (buffer: ArrayBuffer, start: number, length: number): number => {
  const str = parseString(buffer, start, length);
  return parseFloat(str);
};

export const parseEDF = async (file: File): Promise<ParsedEDF> => {
  const buffer = await file.arrayBuffer();
  const header: EDFHeader = {
    version: parseString(buffer, 0, 8),
    patientId: parseString(buffer, 8, 80),
    recordId: parseString(buffer, 88, 80),
    startDate: parseString(buffer, 168, 8),
    startTime: parseString(buffer, 176, 8),
    headerBytes: parseNumber(buffer, 184, 8),
    reserved: parseString(buffer, 192, 44),
    numDataRecords: parseNumber(buffer, 236, 8),
    recordDuration: parseNumber(buffer, 244, 8),
    numSignals: parseNumber(buffer, 252, 4),
  };

  const ns = header.numSignals;
  const signals: SignalHeader[] = [];
  let curr = 256;

  const readSignalProps = (length: number) => {
    const props = [];
    for (let i = 0; i < ns; i++) {
      props.push(parseString(buffer, curr + (i * length), length));
    }
    curr += ns * length;
    return props;
  };

  const readSignalNums = (length: number) => {
    const props = [];
    for (let i = 0; i < ns; i++) {
      props.push(parseFloat(parseString(buffer, curr + (i * length), length)));
    }
    curr += ns * length;
    return props;
  };

  const labels = readSignalProps(16);
  const transducers = readSignalProps(80);
  const dimensions = readSignalProps(8);
  const physMins = readSignalNums(8);
  const physMaxs = readSignalNums(8);
  const digMins = readSignalNums(8);
  const digMaxs = readSignalNums(8);
  const prefilterings = readSignalProps(80);
  const samplesPerRecords = readSignalNums(8);
  curr += ns * 32;

  for (let i = 0; i < ns; i++) {
    signals.push({
      label: labels[i],
      transducerType: transducers[i],
      physicalDimension: dimensions[i],
      physicalMin: physMins[i],
      physicalMax: physMaxs[i],
      digitalMin: digMins[i],
      digitalMax: digMaxs[i],
      prefiltering: prefilterings[i],
      samplesPerRecord: samplesPerRecords[i],
    });
  }

  const dataStart = header.headerBytes;
  const dataView = new DataView(buffer);
  const channelData: Float32Array[] = signals.map(sig => 
    new Float32Array(header.numDataRecords * sig.samplesPerRecord)
  );

  let bufferOffset = dataStart;
  const scales = signals.map(s => (s.physicalMax - s.physicalMin) / (s.digitalMax - s.digitalMin));
  const polyB = signals.map((s, i) => s.physicalMin - (s.digitalMin * scales[i]));

  for (let r = 0; r < header.numDataRecords; r++) {
    for (let s = 0; s < ns; s++) {
      const sig = signals[s];
      const sampleCount = sig.samplesPerRecord;
      const targetArray = channelData[s];
      const startIdx = r * sampleCount;
      const scale = scales[s];
      const bias = polyB[s];

      for (let k = 0; k < sampleCount; k++) {
        const raw = dataView.getInt16(bufferOffset, true);
        targetArray[startIdx + k] = (raw * scale) + bias;
        bufferOffset += 2;
      }
    }
  }

  return { header, signals, data: channelData, totalDuration: header.numDataRecords * header.recordDuration };
};

export const parseCSV = (csvContent: string): { results: AnalysisResult[], metadata: CSVMetadata } => {
  const lines = csvContent.trim().split('\n');
  const results: AnalysisResult[] = [];
  
  let metadata: CSVMetadata = { gender: '', age: '', fileStart: '' };

  const timeToSeconds = (timeStr: string): number => {
    const parts = timeStr.split(':');
    if (parts.length < 3) return 0;
    const hours = parseInt(parts[0]) || 0;
    const minutes = parseInt(parts[1]) || 0;
    const seconds = parseInt(parts[2]) || 0;
    const millis = parseInt(parts[3]) || 0;
    return hours * 3600 + minutes * 60 + seconds + millis / 1000;
  };

  let fileStartSec = 0;
  if (lines.length > 1) {
    const firstDataLine = lines[1].split(',').map(f => f.trim().replace(/^"|"$/g, ''));
    // Capture metadata from the very first data row
    metadata = {
      gender: firstDataLine[0],
      age: firstDataLine[1],
      fileStart: firstDataLine[2]
    };
    if (metadata.fileStart) fileStartSec = timeToSeconds(metadata.fileStart);
  }

  for (let i = 1; i < lines.length; i++) {
    const line = lines[i];
    if (!line.trim()) continue;
    const fields = line.split(',').map(f => f.trim().replace(/^"|"$/g, ''));
    const [ , , , startTime, endTime, channelName, comment] = fields;
    
    if (!startTime || !endTime || !channelName) continue;
    const startSec = timeToSeconds(startTime) - fileStartSec;
    const endSec = timeToSeconds(endTime) - fileStartSec;
    
    results.push({
      window_id: `eeg_event_${i - 1}`,
      classification: 'abnormal',
      channel_name: channelName,
      global_start_time_sec: startSec,
      global_end_time_sec: endSec,
      label: comment || 'Detected',
      event: [{ confidence: 0.95, region_start_time_sec: startSec, region_end_time_sec: endSec }]
    });
  }
  return { results: results.sort((a, b) => a.global_start_time_sec - b.global_start_time_sec), metadata };
};