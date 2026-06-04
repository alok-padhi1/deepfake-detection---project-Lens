// app/(dashboard)/submit/page.tsx
"use client";
import React, { useState } from "react";
import { useRouter } from "next/navigation";
import { useDropzone } from "react-dropzone";
import { UploadCloud, File, AlertCircle, RefreshCw } from "lucide-react";
import { useSubmitMedia } from "@/lib/hooks";

export default function SubmitMediaPage() {
  const router = useRouter();
  const [file, setFile] = useState<File | null>(null);
  const [priority, setPriority] = useState("MEDIUM");
  const [notes, setNotes] = useState("");
  const [uploadProgress, setUploadProgress] = useState(0);
  const [isUploading, setIsUploading] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");

  const submitMutation = useSubmitMedia();

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    maxSize: 500 * 1024 * 1024, // 500MB limit
    multiple: false,
    onDrop: (acceptedFiles, rejectedFiles) => {
      setErrorMessage("");
      if (rejectedFiles.length > 0) {
        setErrorMessage("File rejected. Maximum size is 500MB.");
        return;
      }
      if (acceptedFiles.length > 0) {
        setFile(acceptedFiles[0]);
      }
    }
  });

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) return;
    
    setIsUploading(true);
    setUploadProgress(10);
    
    // Simulate multipart chunk upload progress bar
    const interval = setInterval(() => {
      setUploadProgress((prev) => {
        if (prev >= 90) {
          clearInterval(interval);
          return 90;
        }
        return prev + 10;
      });
    }, 300);

    try {
      const res = await submitMutation.mutateAsync({
        file,
        metadata: { priority, analyst_notes: notes }
      });
      
      clearInterval(interval);
      setUploadProgress(100);
      
      setTimeout(() => {
        router.push(`/cases/${res.case_id}`);
      }, 500);
    } catch (e: any) {
      clearInterval(interval);
      setIsUploading(false);
      setErrorMessage("Analysis request failed. Please check backend Gateway logs.");
    }
  };

  return (
    <div className="max-w-3xl mx-auto space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-white font-mono">MEDIA ANALYSIS SUBMISSION</h1>
        <p className="text-xs text-gray-400">Ingest images, audio, or video files directly into LENS forensics pipeline.</p>
      </div>

      <form onSubmit={handleSubmit} className="bg-surface border border-border p-6 rounded-xl space-y-6">
        
        {/* Dropzone area */}
        <div 
          {...getRootProps()} 
          className={`border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-all ${isDragActive ? "border-forensic-blue bg-forensic-blue/5" : "border-border hover:border-gray-500"}`}
        >
          <input {...getInputProps()} />
          <UploadCloud size={48} className="mx-auto text-gray-500 mb-4" />
          {file ? (
            <div className="space-y-1">
              <p className="text-sm font-semibold text-white">{file.name}</p>
              <p className="text-xs text-gray-400">{(file.size / (1024 * 1024)).toFixed(2)} MB</p>
            </div>
          ) : (
            <div className="space-y-1 text-sm text-gray-400">
              <p className="text-white font-semibold">Drag &amp; drop media file here, or click to select</p>
              <p className="text-xs">Supported formats: JPEG, PNG, WEBP, TIFF, MP4, MOV, MKV, WAV, MP3, FLAC</p>
            </div>
          )}
        </div>

        {errorMessage && (
          <div className="bg-red-500/10 border border-red-500/20 text-red-500 p-3 rounded-lg flex items-center gap-2 text-sm">
            <AlertCircle size={16} />
            {errorMessage}
          </div>
        )}

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <label className="block text-xs uppercase text-gray-500 tracking-wider mb-2">Priority Level</label>
            <select 
              value={priority}
              onChange={(e) => setPriority(e.target.value)}
              className="w-full bg-background border border-border rounded-lg p-2.5 text-sm text-white focus:outline-none focus:border-forensic-blue"
            >
              <option value="LOW">LOW (Standard Batch)</option>
              <option value="MEDIUM">MEDIUM (Standard Queue)</option>
              <option value="HIGH">HIGH (Expedited Queue)</option>
              <option value="CRITICAL">CRITICAL (Immediate Cluster Target)</option>
            </select>
          </div>

          <div>
            <label className="block text-xs uppercase text-gray-500 tracking-wider mb-2">Analyst Reference Notes</label>
            <input 
              type="text"
              value={notes}
              onChange={(e) => setNotes(e.target.value)}
              placeholder="e.g. Case #9124 - Target video speech matching"
              className="w-full bg-background border border-border rounded-lg p-2.5 text-sm text-white focus:outline-none focus:border-forensic-blue"
            />
          </div>
        </div>

        {isUploading && (
          <div className="space-y-2">
            <div className="flex justify-between text-xs font-mono text-gray-400">
              <span className="flex items-center gap-1.5"><RefreshCw size={12} className="animate-spin" /> Stream Uploading to MinIO...</span>
              <span>{uploadProgress}%</span>
            </div>
            <div className="w-full bg-background h-2 rounded-full overflow-hidden">
              <div className="bg-forensic-blue h-full transition-all duration-300" style={{ width: `${uploadProgress}%` }} />
            </div>
          </div>
        )}

        <button
          type="submit"
          disabled={!file || isUploading}
          className="w-full bg-forensic-blue hover:bg-blue-600 disabled:bg-gray-700 disabled:text-gray-400 text-white font-semibold py-3 rounded-lg transition-colors font-mono tracking-wider"
        >
          INITIATE FORENSIC ANALYSIS
        </button>

      </form>
    </div>
  );
}
