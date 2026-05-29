import { jsPDF } from 'jspdf';
import autoTable from 'jspdf-autotable';
import { GoogleGenerativeAI } from "@google/generative-ai";
import { AnalysisResult, ParsedEDF } from '../types';

// Use your new key here
const GEMINI_API_KEY = "AIzaSyDyOVFGxUHiEwyR-ilAVElGARtsKE5sjjs"; 
const genAI = new GoogleGenerativeAI(GEMINI_API_KEY);

export const generateEEGReport = async (
  fileName: string,
  data: ParsedEDF,
  localEvents: AnalysisResult[],
  formatTime: (s: number) => string
) => {
  try {
    // Force the correct model string and ensure no hidden characters
    const modelName = "gemini-2.5-flash";
    // Use 'gemini-1.5-flash-latest' to point to the most recent stable version
const model = genAI.getGenerativeModel({ model: "gemini-2.5-flash" });

    const detectionsSummary = localEvents.map(ev => ({
      time: `${formatTime(ev.global_start_time_sec)} - ${formatTime(ev.global_end_time_sec)}`,
      channel: ev.channel_name,
      label: ev.label || 'Unspecified Abnormality'
    }));

    const prompt = `
      As a Clinical Neurophysiologist, interpret these EEG findings:
      File: ${fileName}
      ID: ${data.header.patientId}
      Findings: ${JSON.stringify(detectionsSummary)}
      
      Structure: Clinical Description, Findings, and Impression.
    `;

    console.log(`Requesting ${modelName}...`);
    const result = await model.generateContent(prompt);
    const response = await result.response;
    const text = response.text();

    const doc = new jsPDF();
    const pageWidth = doc.internal.pageSize.getWidth();

    // Styled Header
    doc.setFillColor(15, 23, 42);
    doc.rect(0, 0, pageWidth, 40, 'F');
    doc.setTextColor(255, 255, 255);
    doc.setFontSize(20);
    doc.text("NEUROXPLAIN CLINICAL REPORT", 14, 25);

    // AI Narrative
    doc.setTextColor(40, 40, 40);
    doc.setFontSize(10);
    const splitText = doc.splitTextToSize(text, pageWidth - 28);
    doc.text(splitText, 14, 55);

    // Table of Events
    autoTable(doc, {
      startY: 55 + (splitText.length * 5) + 10,
      head: [['Segment', 'Channel', 'Finding']],
      body: detectionsSummary.map(d => [d.time, d.channel, d.label]),
      headStyles: { fillColor: [79, 70, 229] }
    });

    doc.save(`Report_${fileName.split('.')[0]}.pdf`);
    return true;

  } catch (error: any) {
    console.error("DETAILED REPORT ERROR:", error);
    // Log the specific error message from Google if available
    if (error.message) console.error("Error Message:", error.message);
    throw error;
  }
};