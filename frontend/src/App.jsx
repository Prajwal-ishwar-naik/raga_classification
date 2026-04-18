import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import WaveSurfer from 'wavesurfer.js';
import { 
  Music, 
  Upload, 
  Zap,
  Play, 
  Pause,
  Layers,
  Sparkles,
  ChevronRight,
  TrendingUp,
  Files
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

const SpectrogramView = ({ data }) => {
  const canvasRef = useRef(null);

  useEffect(() => {
    if (canvasRef.current && data && data.length > 0) {
      const ctx = canvasRef.current.getContext('2d');
      const width = data[0].length;
      const height = data.length;
      canvasRef.current.width = width;
      canvasRef.current.height = height;
      
      const imgData = ctx.createImageData(width, height);
      for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
          const idx = (y * width + x) * 4;
          const val = data[height - 1 - y][x];
          const v = val / 255.0;
          imgData.data[idx] = v < 0.2 ? v * 50 : (v < 0.5 ? 100 + (v-0.2)*400 : 255);
          imgData.data[idx + 1] = v > 0.4 ? (v - 0.4) * 1.5 * 255 : 0;
          imgData.data[idx + 2] = v > 0.7 ? (v - 0.7) * 3 * 255 : (v < 0.3 ? v * 100 : 0);
          imgData.data[idx + 3] = 255;
        }
      }
      ctx.putImageData(imgData, 0, 0);
    }
  }, [data]);

  return (
    <div className="spectrogram-wrapper">
      <div className="freq-axis">
        <span>High</span>
        <span>Freq</span>
        <span>Low</span>
      </div>
      <div className="spectrogram-card" style={{ flex: 1 }}>
        <canvas ref={canvasRef} className="spectrogram-canvas" />
      </div>
    </div>
  );
};

const BulkResultsView = ({ data, files }) => {
  const dayCount = data.filter(r => r.prediction.includes('Day')).length;
  const nightCount = data.filter(r => r.prediction.includes('Night')).length;
  const [expandedIndex, setExpandedIndex] = useState(null);
  const [viewMode, setViewMode] = useState('narrative'); // 'narrative' or 'spectrogram'

  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="bulk-view">
      <div className="bulk-stats">
        <div className="stat-card card">
          <span className="label">Day Raags</span>
          <div className="stat-val">{dayCount}</div>
        </div>
        <div className="stat-card card">
          <span className="label">Night Raags</span>
          <div className="stat-val">{nightCount}</div>
        </div>
      </div>

      <div className="bulk-table card">
        <div className="table-header">
          <span>Raga Sample</span>
          <span>Category</span>
          <span>Decision</span>
        </div>
        <div className="table-body">
          {data.map((item, i) => (
            <React.Fragment key={i}>
              <div 
                className={`table-row ${expandedIndex === i ? 'expanded' : ''}`}
              >
                <div className="row-file-box" style={{ flexDirection: 'column', alignItems: 'flex-start', gap: '6px', paddingRight: '1rem' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                    <Music size={14} className="text-primary" />
                    <span className="row-file" style={{ paddingRight: 0 }}>{item.filename}</span>
                  </div>
                  {files && files.find(f => f.name === item.filename) && (
                    <audio 
                      src={URL.createObjectURL(files.find(f => f.name === item.filename))} 
                      controls 
                      style={{ height: '28px', width: '100%', maxWidth: '240px' }} 
                    />
                  )}
                </div>
                <span className={`row-pred ${item.prediction.includes('Day') ? 'day' : 'night'}`}>
                  {item.prediction}
                </span>
                <div className="action-buttons" style={{ display: 'flex', gap: '8px' }}>
                  <button 
                    className={`summarize-btn ${expandedIndex === i && viewMode === 'narrative' ? 'active' : ''}`}
                    onClick={() => { setExpandedIndex(expandedIndex === i && viewMode === 'narrative' ? null : i); setViewMode('narrative'); }}
                    style={{ background: expandedIndex === i && viewMode === 'narrative' ? 'var(--primary)' : 'rgba(255,255,255,0.05)' }}
                  >
                    Explanation
                  </button>
                  <button 
                    className={`summarize-btn ${expandedIndex === i && viewMode === 'spectrogram' ? 'active' : ''}`}
                    onClick={() => { setExpandedIndex(expandedIndex === i && viewMode === 'spectrogram' ? null : i); setViewMode('spectrogram'); }}
                    style={{ background: expandedIndex === i && viewMode === 'spectrogram' ? 'var(--primary)' : 'rgba(255,255,255,0.05)' }}
                  >
                    Spectrogram Graph
                  </button>
                </div>
              </div>
              <AnimatePresence>
                {expandedIndex === i && (
                  <motion.div 
                    initial={{ height: 0, opacity: 0 }} 
                    animate={{ height: 'auto', opacity: 1 }} 
                    exit={{ height: 0, opacity: 0 }}
                    className="row-expansion"
                  >
                    <div className="expansion-inner">
                      {viewMode === 'narrative' ? (
                        <div className="hybrid-explanation">
                          <div className="label"><Sparkles size={14} style={{display:'inline', marginRight:'4px'}}/> AI Reasoned Analysis</div>
                          <p className="narrative-text-small">{item.narrative}</p>
                        </div>
                      ) : (
                        <div className="visual-block" style={{ padding: '1rem' }}>
                          <span className="label" style={{marginBottom: '0.5rem', display: 'block'}}>Acoustic Spectrogram Signature</span>
                          {item.spectrogram ? <SpectrogramView data={item.spectrogram} /> : <p className="narrative-text-small">Spectrogram data unavailable for this segment.</p>}
                        </div>
                      )}
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </React.Fragment>
          ))}
        </div>
      </div>
    </motion.div>
  );
};

const App = () => {
  const [files, setFiles] = useState([]);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [bulkResults, setBulkResults] = useState(null);
  const [playing, setPlaying] = useState(false);
  const [error, setError] = useState(null);
  
  const waveformRef = useRef(null);
  const wavesurfer = useRef(null);

  useEffect(() => {
    if (waveformRef.current && files.length === 1 && !wavesurfer.current) {
      wavesurfer.current = WaveSurfer.create({
        container: waveformRef.current,
        waveColor: '#b08d4822',
        progressColor: '#b08d48',
        height: 60,
        barWidth: 3,
        normalize: true,
      });
      wavesurfer.current.on('finish', () => setPlaying(false));
      wavesurfer.current.load(URL.createObjectURL(files[0]));
    }
    return () => wavesurfer.current?.destroy();
  }, [files]);

  const handleUpload = async () => {
    if (files.length === 0) return;
    setLoading(true);
    setResult(null);
    setBulkResults(null);
    setError(null);

    const formData = new FormData();
    if (files.length === 1) {
      formData.append('file', files[0]);
      try {
        const response = await axios.post('/api/classify', formData);
        setResult(response.data);
      } catch (err) {
        setError('Inference Error: ' + (err.response?.data?.detail || 'Server Down'));
      }
    } else {
      Array.from(files).forEach(f => formData.append('files', f));
      try {
        const response = await axios.post('/api/classify_bulk', formData);
        setBulkResults(response.data.results);
      } catch (err) {
        const detail = err.response?.data?.detail || err.message;
        setError(`Bulk Analysis Failed: ${detail}`);
      }
    }
    setLoading(false);
  };

  return (
    <div className="container">
      <header>
        <p className="subtitle">Enterprise Neural Intelligence</p>
        <h1 className="main-title">RAGA VISION <span className="neural-text">BULK EDITION</span></h1>
      </header>

      <div className="main-layout">
        <section className="card glass-card">
          {files.length === 0 ? (
            <div className="upload-zone" onClick={() => document.getElementById('audio-input').click()}>
              <Files className="upload-icon" size={60} />
              <h3 className="upload-title">Bulk Audio Input</h3>
              <p className="upload-subtitle">Select multiple WAV, MP3, or Opus Raag recordings</p>
              <input 
                id="audio-input" 
                type="file" 
                multiple 
                onChange={(e) => setFiles(Array.from(e.target.files))} 
                style={{ display: 'none' }} 
              />
            </div>
          ) : (
            <div className="analyzer-view">
              <div className="file-info">
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.8rem' }}>
                  <TrendingUp color="var(--primary)" size={18} />
                  <span style={{ fontWeight: 700 }}>{files.length} Samples Staged for Analysis</span>
                </div>
                <button className="reset-btn" onClick={() => { setFiles([]); setResult(null); setBulkResults(null); }}>Clear Session</button>
              </div>

              {files.length === 1 && <div className="waveform-box" ref={waveformRef}></div>}

              <div className="action-row">
                {files.length === 1 && (
                  <button className="play-btn" onClick={() => { wavesurfer.current.playPause(); setPlaying(!playing); }}>
                    {playing ? <Pause size={24} /> : <Play size={24} />}
                  </button>
                )}
                {!result && !bulkResults && (
                  <button className="analyze-btn" onClick={handleUpload} disabled={loading}>
                    {loading ? 'Processing Neural Queue...' : `Analyze ${files.length} Input(s)`}
                  </button>
                )}
              </div>
              {error && <div className="error-box">{error}</div>}
            </div>
          )}
        </section>

        <AnimatePresence>
          {result && (
            <motion.div initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }} className="result-container single">
              <div className="neural-intel-box card">
                <div className="result-header">
                  <div>
                    <span className="label">Neural Prediction</span>
                    <h2 className="prediction-display">{result.neural_prediction}</h2>
                  </div>
                  <div className="confidence-ring">
                    <span className="conf-value">{(result.neural_confidence * 100).toFixed(0)}%</span>
                    <span className="conf-label">Confidence</span>
                  </div>
                </div>

                <div className="neural-visuals">
                  <div className="visual-block">
                    <span className="label">Neural Feature Extraction</span>
                    <SpectrogramView data={result.spectrogram} />
                  </div>
                  
                  <div className="insight-block hybrid">
                    <div className="hybrid-badge">
                      <Sparkles size={14} />
                      <span>Hybrid Intel Active</span>
                    </div>

                    <div className="insight-item primary">
                      <Music size={20} />
                      <div>
                        <span className="label">Identified Raag</span>
                        <h3>{result.detected_raag}</h3>
                      </div>
                    </div>

                    <div className="insight-item">
                      <Zap size={16} />
                      <span>{result.neural_mood} Neural Mood Detected</span>
                    </div>

                    <div className="swara-chips">
                      {result.metadata.swaras.map((s, idx) => (
                        <span key={idx} className="swara-chip">{s}</span>
                      ))}
                    </div>

                    <div className="logic-message">
                      {result.report[0] || "Aligned with traditional musicology."}
                    </div>
                  </div>
                </div>

                <div className="narrative-section card">
                  <div className="label" style={{ marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <Sparkles size={16} /> Reasoning Narrative
                  </div>
                  <p className="narrative-text">{result.narrative}</p>
                </div>
              </div>
            </motion.div>
          )}

          {bulkResults && <BulkResultsView data={bulkResults} files={files} />}
        </AnimatePresence>
      </div>
    </div>
  );
};

export default App;
