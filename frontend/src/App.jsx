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
  Files,
  Heart,
  Focus,
  Activity,
  CheckCircle2,
  AlertCircle
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
const API_BASE = "http://127.0.0.1:8000";
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
  const [viewMode, setViewMode] = useState('narrative'); // 'narrative', 'spectrogram', or 'therapy'

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
                    Visual Analytics
                  </button>
                  <button 
                    className={`summarize-btn ${expandedIndex === i && viewMode === 'therapy' ? 'active' : ''}`}
                    onClick={() => { setExpandedIndex(expandedIndex === i && viewMode === 'therapy' ? null : i); setViewMode('therapy'); }}
                    style={{ background: expandedIndex === i && viewMode === 'therapy' ? 'var(--accent)' : 'rgba(245, 158, 11, 0.1)', color: expandedIndex === i && viewMode === 'therapy' ? '#0f1115' : 'var(--primary)' }}
                  >
                    Therapy Analysis
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
                        <div className="hybrid-explanation" style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                          <div>
                            <div className="label"><Sparkles size={14} style={{display:'inline', marginRight:'4px'}}/> AI Reasoned Analysis</div>
                            <p className="narrative-text-small">{item.narrative}</p>
                          </div>
                          {item.detailed_features && (
                            <div style={{ transform: 'scale(0.95)', transformOrigin: 'top left', marginTop: '-1rem' }}>
                              <DetailedFeatureAnalysis features={item.detailed_features} />
                            </div>
                          )}
                        </div>
                      ) : viewMode === 'spectrogram' ? (
                        <div className="visual-block" style={{ padding: '1rem' }}>
                          <span className="label" style={{marginBottom: '0.5rem', display: 'block'}}>Acoustic Spectrogram Signature</span>
                          {item.spectrogram ? <SpectrogramView data={item.spectrogram} /> : <p className="narrative-text-small">Spectrogram data unavailable for this segment.</p>}
                          
                          {item.image_url && (
                            <div className="analysis-visualization" style={{ marginTop: '1.5rem' }}>
                              <span className="label" style={{marginBottom: '0.5rem', display: 'block'}}>Visual Analysis Dashboard</span>
                              <div className="image-container" style={{ textAlign: 'center', background: '#111', borderRadius: '8px', padding: '0.5rem' }}>
                                <img src={item.image_url} alt="Raga Analysis Dashboard" style={{ maxWidth: '100%', height: 'auto', borderRadius: '4px', display: 'block', margin: '0 auto' }} />
                              </div>
                            </div>
                          )}
                          
                          {item.detailed_features && (
                            <div style={{ transform: 'scale(0.95)', transformOrigin: 'top left', marginTop: '1rem' }}>
                              <DetailedFeatureAnalysis features={item.detailed_features} />
                            </div>
                          )}
                        </div>
                      ) : (
                        <div className="therapy-block" style={{ padding: '1rem' }}>
                          <span className="label" style={{marginBottom: '1rem', display: 'block'}}>🧘 Wellness & Therapy Profile</span>
                          {item.therapy ? (
                            <div className="bulk-therapy-grid" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1.5rem' }}>
                              <div className="scores-sub-block">
                                <div className="score-item" style={{marginBottom: '0.8rem'}}>
                                  <div style={{display:'flex', justifyContent:'space-between', fontSize:'0.8rem', fontWeight:700}}><span>Calm</span><span>{item.therapy.therapy_scores.calm_score}/10</span></div>
                                  <div className="progress-bg"><div className="progress-fill calm" style={{width:`${item.therapy.therapy_scores.calm_score*10}%`}}></div></div>
                                </div>
                                <div className="score-item" style={{marginBottom: '0.8rem'}}>
                                  <div style={{display:'flex', justifyContent:'space-between', fontSize:'0.8rem', fontWeight:700}}><span>Energy</span><span>{item.therapy.therapy_scores.energy_score}/10</span></div>
                                  <div className="progress-bg"><div className="progress-fill energy" style={{width:`${item.therapy.therapy_scores.energy_score*10}%`}}></div></div>
                                </div>
                                <div className="score-item">
                                  <div style={{display:'flex', justifyContent:'space-between', fontSize:'0.8rem', fontWeight:700}}><span>Focus</span><span>{item.therapy.therapy_scores.focus_score}/10</span></div>
                                  <div className="progress-bg"><div className="progress-fill focus" style={{width:`${item.therapy.therapy_scores.focus_score*10}%`}}></div></div>
                                </div>
                              </div>
                              <div className="rec-sub-block">
                                <div style={{background:'rgba(255,255,255,0.05)', padding:'0.8rem', borderRadius:'8px', border:'1px solid var(--border)'}}>
                                  <div style={{fontSize:'0.7rem', color:'var(--text-dim)', marginBottom:'0.2rem'}}>Primary Recommendation</div>
                                  <div style={{fontWeight:800, color:'var(--primary)', fontSize:'0.9rem'}}>{item.therapy.recommendation.primary}</div>
                                </div>
                                <ul style={{marginTop:'0.8rem', fontSize:'0.8rem', paddingLeft:'1rem', color:'var(--text-main)'}}>
                                  {item.therapy.explanation.slice(0,2).map((e, idx) => <li key={idx}>{e}</li>)}
                                </ul>
                              </div>
                            </div>
                          ) : <p className="narrative-text-small">Therapy data not available.</p>}
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

const DetailedFeatureAnalysis = ({ features }) => {
  if (!features) return null;
  return (
    <div className="detailed-analysis-section card" style={{ marginTop: '2rem', background: 'rgba(255,255,255,0.02)', border: '1px solid var(--border)' }}>
      <div style={{ textAlign: 'center', marginBottom: '2rem', borderBottom: '1px solid rgba(176, 141, 72, 0.2)', paddingBottom: '1rem' }}>
        <h3 style={{ color: 'var(--accent)', letterSpacing: '0.15em', fontWeight: 800, fontSize: '1.2rem' }}>FEATURE ANALYSIS</h3>
      </div>
      
      <div className="feature-list" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '2rem' }}>
        
        {/* Swaras */}
        <div className="feature-item">
          <h4 style={{ color: 'var(--primary)', marginBottom: '0.8rem', display: 'flex', alignItems: 'center', gap: '8px', fontSize: '1rem' }}>🎵 Swaras</h4>
          <div style={{ color: 'var(--text-main)', fontSize: '0.85rem', lineHeight: '1.8' }}>
            <div><span style={{ fontWeight: 700, color: 'var(--accent)' }}>Detected:</span> {features.swaras.detected}</div>
            <div><span style={{ fontWeight: 700, color: 'var(--accent)' }}>Unique:</span> {features.swaras.unique}</div>
            <div><span style={{ fontWeight: 700, color: 'var(--accent)' }}>Most Frequent:</span> {features.swaras.most_frequent}</div>
          </div>
        </div>

        {/* Arohana-Avarohana */}
        <div className="feature-item">
          <h4 style={{ color: '#4db8ff', marginBottom: '0.8rem', display: 'flex', alignItems: 'center', gap: '8px', fontSize: '1rem' }}>📈 Arohana-Avarohana</h4>
          <div style={{ color: 'var(--text-main)', fontSize: '0.85rem', lineHeight: '1.8' }}>
            <div><span style={{ fontWeight: 700, color: 'var(--accent)' }}>Arohana:</span> {features.arohana_avarohana.arohana}</div>
            <div><span style={{ fontWeight: 700, color: 'var(--accent)' }}>Avarohana:</span> {features.arohana_avarohana.avarohana}</div>
          </div>
        </div>

        {/* Pakad */}
        <div className="feature-item">
          <h4 style={{ color: '#ff4757', marginBottom: '0.8rem', display: 'flex', alignItems: 'center', gap: '8px', fontSize: '1rem' }}>🎯 Pakad</h4>
          <div style={{ color: 'var(--text-main)', fontSize: '0.85rem', lineHeight: '1.8' }}>
            {features.pakad.map((p, idx) => p ? <div key={idx}><span style={{ fontWeight: 700, color: 'var(--accent)' }}>{idx + 1}.</span> {p}</div> : null)}
          </div>
        </div>

        {/* Gamakas */}
        <div className="feature-item">
          <h4 style={{ color: '#7158e2', marginBottom: '0.8rem', display: 'flex', alignItems: 'center', gap: '8px', fontSize: '1rem' }}>🎶 Gamakas</h4>
          <div style={{ color: 'var(--text-main)', fontSize: '0.85rem', lineHeight: '1.8' }}>
            <div><span style={{ fontWeight: 700, color: 'var(--accent)' }}>Oscillations:</span> {features.gamakas.oscillations}</div>
            <div><span style={{ fontWeight: 700, color: 'var(--accent)' }}>Slides detected:</span> {features.gamakas.slides}</div>
            <div><span style={{ fontWeight: 700, color: 'var(--accent)' }}>Avg Pitch Variation:</span> {features.gamakas.avg_var}</div>
          </div>
        </div>

        {/* Vadi-Samvadi */}
        <div className="feature-item">
          <h4 style={{ color: '#fbc531', marginBottom: '0.8rem', display: 'flex', alignItems: 'center', gap: '8px', fontSize: '1rem' }}>⭐ Vadi-Samvadi</h4>
          <div style={{ color: 'var(--text-main)', fontSize: '0.85rem', lineHeight: '1.8' }}>
            <div><span style={{ fontWeight: 700, color: 'var(--accent)' }}>Vadi:</span> {features.vadi_samvadi.vadi}</div>
            <div><span style={{ fontWeight: 700, color: 'var(--accent)' }}>Samvadi:</span> {features.vadi_samvadi.samvadi}</div>
          </div>
        </div>

        {/* Pitch Range */}
        <div className="feature-item">
          <h4 style={{ color: '#00cec9', marginBottom: '0.8rem', display: 'flex', alignItems: 'center', gap: '8px', fontSize: '1rem' }}>📊 Pitch Range</h4>
          <div style={{ color: 'var(--text-main)', fontSize: '0.85rem', lineHeight: '1.8' }}>
            <div><span style={{ fontWeight: 700, color: 'var(--accent)' }}>Min:</span> {features.pitch_range.min}, <span style={{ fontWeight: 700, color: 'var(--accent)' }}>Max:</span> {features.pitch_range.max}</div>
            <div><span style={{ fontWeight: 700, color: 'var(--accent)' }}>Range:</span> {features.pitch_range.range}</div>
          </div>
        </div>

        {/* Note Transitions */}
        <div className="feature-item">
          <h4 style={{ color: '#e84393', marginBottom: '0.8rem', display: 'flex', alignItems: 'center', gap: '8px', fontSize: '1rem' }}>🔄 Note Transitions</h4>
          <div style={{ color: 'var(--text-main)', fontSize: '0.85rem', lineHeight: '1.8' }}>
            <div><span style={{ fontWeight: 700, color: 'var(--accent)' }}>Most Common:</span> {features.transitions.most_common}</div>
          </div>
        </div>

        {/* Tempo & Structure */}
        <div className="feature-item">
          <h4 style={{ color: '#d63031', marginBottom: '0.8rem', display: 'flex', alignItems: 'center', gap: '8px', fontSize: '1rem' }}>⏱️ Timing & Structure</h4>
          <div style={{ color: 'var(--text-main)', fontSize: '0.85rem', lineHeight: '1.8' }}>
            <div><span style={{ fontWeight: 700, color: 'var(--accent)' }}>BPM:</span> {features.tempo.bpm}</div>
            <div style={{ whiteSpace: 'pre-wrap', marginTop: '0.5rem' }}><span style={{ fontWeight: 700, color: 'var(--accent)' }}>Structure:</span><br/>{features.structure}</div>
          </div>
        </div>

        {/* Timbre */}
        <div className="feature-item">
          <h4 style={{ color: '#636e72', marginBottom: '0.8rem', display: 'flex', alignItems: 'center', gap: '8px', fontSize: '1rem' }}>🎧 Timbre</h4>
          <div style={{ color: 'var(--text-main)', fontSize: '0.85rem', lineHeight: '1.8' }}>
            <div><span style={{ fontWeight: 700, color: 'var(--accent)' }}>MFCC Mean:</span> {features.timbre.mfcc_mean}</div>
            <div><span style={{ fontWeight: 700, color: 'var(--accent)' }}>Spectral Centroid:</span> {features.timbre.centroid}</div>
            <div><span style={{ fontWeight: 700, color: 'var(--accent)' }}>ZCR:</span> {features.timbre.zcr}</div>
          </div>
        </div>

        {/* Advanced Analytics */}
        {features.advanced_analytics && (
          <div className="feature-item">
            <h4 style={{ color: '#00b894', marginBottom: '0.8rem', display: 'flex', alignItems: 'center', gap: '8px', fontSize: '1rem' }}>🧠 AI Confidence Reasoning</h4>
            <div style={{ color: 'var(--text-main)', fontSize: '0.85rem', lineHeight: '1.8' }}>
              <div><span style={{ fontWeight: 700, color: 'var(--accent)' }}>Sa Stability:</span> {(features.advanced_analytics.sa_stability || 0).toFixed(2)}</div>
              <div><span style={{ fontWeight: 700, color: 'var(--accent)' }}>Nyas Swaras:</span> {(features.advanced_analytics.nyas_swaras || []).join(', ') || 'None'}</div>
              <div style={{ marginTop: '0.5rem' }}>
                <span style={{ fontWeight: 700, color: 'var(--accent)' }}>Confidence Explanation:</span>
                <ul style={{ paddingLeft: '1.2rem', marginTop: '0.2rem', marginBottom: 0 }}>
                  {(features.advanced_analytics.confidence_reason || []).map((reason, idx) => (
                    <li key={idx}>{reason}</li>
                  ))}
                </ul>
              </div>
            </div>
          </div>
        )}

      </div>
    </div>
  );
};

const App = () => {
  const [files, setFiles] = useState([]);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [bulkResults, setBulkResults] = useState(null);
  const [playing, setPlaying] = useState(false);
  const [error, setError] = useState(null);
  const [showTherapy, setShowTherapy] = useState(false);
  
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
        const response = await axios.post(`${API_BASE}/classify`, formData);
        setResult(response.data);
      } catch (err) {
        setError('Inference Error: ' + (err.response?.data?.detail || 'Server Down'));
      }
    } else {
      Array.from(files).forEach(f => formData.append('files', f));
      try {
        const response = await axios.post(`${API_BASE}/classify_bulk`, formData);
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
            <motion.div 
              initial={{ opacity: 0, y: 20 }} 
              animate={{ opacity: 1, y: 0 }} 
              className="result-container"
            >
              <div className="card neural-intel-box">
                
                {/* --- NEW SECTION: THERAPY ANALYSIS (TOGGLEABLE) --- */}
                <AnimatePresence>
                  {showTherapy && result.therapy && (
                    <motion.div 
                      initial={{ opacity: 0, height: 0, marginBottom: 0 }} 
                      animate={{ opacity: 1, height: 'auto', marginBottom: '2rem' }} 
                      exit={{ opacity: 0, height: 0, marginBottom: 0 }}
                      className="therapy-analysis-card"
                      style={{ 
                        overflow: 'hidden',
                        padding: '2rem', 
                        background: 'rgba(176, 141, 72, 0.05)', 
                        borderRadius: '30px', 
                        border: '1px solid var(--primary)' 
                      }}
                    >
                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '1.5rem' }}>
                        <span style={{ fontSize: '1.5rem' }}>🧘</span>
                        <h2 style={{ margin: 0, color: 'var(--accent)', fontSize: '1.75rem' }}>Therapy Analysis</h2>
                      </div>

                      <div className="therapy-scores-row" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '2rem', marginBottom: '2rem' }}>
                        <div className="score-block">
                          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem', fontWeight: 800, color: 'var(--accent)' }}>
                            <span>Calm Score</span>
                            <span>{result.therapy.therapy_scores.calm_score}/10</span>
                          </div>
                          <div className="progress-bg"><div className="progress-fill calm" style={{ width: `${result.therapy.therapy_scores.calm_score * 10}%` }}></div></div>
                        </div>
                        <div className="score-block">
                          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem', fontWeight: 800, color: 'var(--accent)' }}>
                            <span>Energy Score</span>
                            <span>{result.therapy.therapy_scores.energy_score}/10</span>
                          </div>
                          <div className="progress-bg"><div className="progress-fill energy" style={{ width: `${result.therapy.therapy_scores.energy_score * 10}%` }}></div></div>
                        </div>
                        <div className="score-block">
                          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem', fontWeight: 800, color: 'var(--accent)' }}>
                            <span>Focus Score</span>
                            <span>{result.therapy.therapy_scores.focus_score}/10</span>
                          </div>
                          <div className="progress-bg"><div className="progress-fill focus" style={{ width: `${result.therapy.therapy_scores.focus_score * 10}%` }}></div></div>
                        </div>
                      </div>

                      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '2rem' }}>
                        <div className="recommendation-panel">
                          <div style={{ marginBottom: '1.5rem' }}>
                            <span className="label" style={{ display: 'block', marginBottom: '0.5rem' }}>Primary Recommendation</span>
                            <div style={{ fontSize: '1.25rem', fontWeight: 900, color: 'var(--primary)', background: 'white', padding: '1rem', borderRadius: '15px', border: '1px solid var(--border)' }}>
                              {result.therapy.recommendation.primary}
                            </div>
                          </div>
                          <div>
                            <span className="label" style={{ display: 'block', marginBottom: '0.5rem' }}>Secondary Recommendations</span>
                            <ul style={{ margin: 0, paddingLeft: '1.25rem', color: 'var(--text-dim)', fontWeight: 600 }}>
                              {result.therapy.recommendation.secondary.map((rec, i) => (
                                <li key={i} style={{ marginBottom: '0.25rem' }}>{rec}</li>
                              ))}
                            </ul>
                          </div>
                        </div>

                        <div className="explanation-panel">
                          <span className="label" style={{ display: 'block', marginBottom: '0.5rem' }}>Therapeutic Explanation</span>
                          <div style={{ background: 'rgba(255,255,255,0.5)', padding: '1.25rem', borderRadius: '15px', border: '1px solid var(--border)' }}>
                            <ul style={{ margin: 0, paddingLeft: '1.25rem', fontSize: '0.9rem', color: 'var(--text-main)', lineHeight: '1.5' }}>
                              {result.therapy.explanation.map((point, i) => (
                                <li key={i} style={{ marginBottom: '0.5rem' }}>{point}</li>
                              ))}
                            </ul>
                          </div>
                        </div>
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>

                <div className="result-header">
                  <div>
                    <span className="label">Neural Prediction</span>
                    <h2 className="prediction-display">{result.neural_prediction}</h2>
                  </div>
                  <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: '1rem' }}>

                    <button 
                      onClick={() => setShowTherapy(!showTherapy)}
                      className="therapy-toggle-btn"
                      style={{
                        padding: '0.6rem 1.2rem',
                        background: showTherapy ? 'var(--accent)' : 'var(--primary)',
                        color: '#0f1115',
                        border: 'none',
                        borderRadius: '10px',
                        fontWeight: '800',
                        fontSize: '0.85rem',
                        cursor: 'pointer',
                        display: 'flex',
                        alignItems: 'center',
                        gap: '0.5rem',
                        transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
                        boxShadow: '0 4px 12px rgba(176, 141, 72, 0.2)'
                      }}
                    >
                      <Heart size={16} fill={showTherapy ? '#0f1115' : 'transparent'} />
                      {showTherapy ? 'Hide Therapy Analysis' : 'View Therapy Analysis'}
                    </button>
                  </div>
                </div>

                {/* Removed from here to place below spectrogram */}

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

                    {result.metadata.advanced_features && result.metadata.advanced_features.most_frequent && (
                      <div className="feature-summary" style={{ marginTop: '1rem', padding: '0.8rem', background: 'rgba(0,0,0,0.2)', borderRadius: '6px' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                          <div>
                            <div style={{ fontSize: '0.8rem', color: '#aaa', marginBottom: '0.2rem' }}>Most Frequent Note</div>
                            <div style={{ fontWeight: 600, color: 'var(--primary)' }}>{result.metadata.advanced_features.most_frequent}</div>
                          </div>
                          <div>
                            <div style={{ fontSize: '0.8rem', color: '#aaa', marginBottom: '0.2rem' }}>Vadi Note</div>
                            <div style={{ fontWeight: 600, color: '#4db8ff' }}>{result.metadata.advanced_features.vadi}</div>
                          </div>
                        </div>
                      </div>
                    )}

                    <div className="logic-message" style={{ marginTop: '1rem' }}>
                      {result.report[0] || "Aligned with traditional musicology."}
                    </div>
                  </div>
                </div>

                {/* --- LEGACY COMPREHENSIVE DASHBOARD (PROMINENT) MOVED HERE --- */}
                {result.image_url && (
                  <div className="analysis-visualization card" style={{ marginTop: '1.5rem', marginBottom: '0rem' }}>
                    <div className="label" style={{ marginBottom: '1rem', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                        <Layers size={16} /> Visual Analysis Dashboard
                      </div>
                      <a href={result.image_url} download={`raga_analysis_${result.detected_raag}.png`} className="download-btn" style={{ padding: '0.4rem 0.8rem', background: 'var(--primary)', color: '#000', borderRadius: '4px', textDecoration: 'none', fontSize: '0.85rem', fontWeight: 'bold' }}>
                        Download PNG
                      </a>
                    </div>
                    <div className="image-container" style={{ textAlign: 'center', background: '#111', borderRadius: '8px', padding: '0.5rem' }}>
                      <img src={result.image_url} alt="Raga Analysis Dashboard" style={{ maxWidth: '100%', height: 'auto', borderRadius: '4px', display: 'block' }} />
                    </div>
                  </div>
                )}


                <div className="narrative-section card" style={{ marginTop: '1.5rem' }}>
                  <div className="label" style={{ marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <Sparkles size={16} /> Reasoning Narrative
                  </div>
                  <p className="narrative-text">{result.narrative}</p>
                </div>


                <DetailedFeatureAnalysis features={result.detailed_features} />

                {/* --- NEW: DETAILED VISUALIZATIONS --- */}
                <div className="detailed-visuals-grid" style={{ marginTop: '1.5rem', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem' }}>
                  {result.pitch_contour_url && (
                    <div className="visual-card card">
                      <span className="label" style={{ marginBottom: '1rem', display: 'block' }}>Pitch Contour Analysis</span>
                      <img src={result.pitch_contour_url} alt="Pitch Contour" style={{ width: '100%', borderRadius: '4px' }} />
                    </div>
                  )}
                  {result.spectrogram_url && (
                    <div className="visual-card card">
                      <span className="label" style={{ marginBottom: '1rem', display: 'block' }}>Spectral Fingerprint</span>
                      <img src={result.spectrogram_url} alt="Spectrogram" style={{ width: '100%', borderRadius: '4px' }} />
                    </div>
                  )}
                </div>

                {/* Disclaimer */}
                <div className="disclaimer-footer" style={{ marginTop: '2rem', padding: '1rem', borderTop: '1px solid rgba(255,255,255,0.05)', textAlign: 'center', opacity: 0.6 }}>
                  <p style={{ fontSize: '0.8rem', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.4rem' }}>
                    <AlertCircle size={12} /> This system provides recommendations based on audio features and is not a medical diagnosis tool.
                  </p>
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
