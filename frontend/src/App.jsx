import { useEffect, useRef, useState } from 'react';
import './index.css';

const API_BASE = 'http://localhost:8000';

const Icons = {
  video: (
    <svg width="1em" height="1em" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="m16 13 5.223 3.482a.5.5 0 0 0 .777-.416V7.934a.5.5 0 0 0-.777-.416L16 11" />
      <rect x="2" y="6" width="14" height="12" rx="2" />
    </svg>
  ),
  upload: (
    <svg width="1em" height="1em" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
      <polyline points="17 8 12 3 7 8" />
      <line x1="12" y1="3" x2="12" y2="15" />
    </svg>
  ),
  camera: (
    <svg width="1em" height="1em" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M14.5 4h-5L7 7H4a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V9a2 2 0 0 0-2-2h-3l-2.5-3z" />
      <circle cx="12" cy="13" r="3" />
    </svg>
  ),
  stop: (
    <svg width="1em" height="1em" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <rect x="6" y="6" width="12" height="12" rx="2" />
    </svg>
  ),
  sparkles: (
    <svg width="1em" height="1em" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M9.937 15.5A2 2 0 0 0 8.5 14.063l-6.135-1.582a.5.5 0 0 1 0-.962L8.5 9.936A2 2 0 0 0 9.937 8.5l1.582-6.135a.5.5 0 0 1 .963 0L14.063 8.5A2 2 0 0 0 15.5 9.937l6.135 1.581a.5.5 0 0 1 0 .964L15.5 14.063a2 2 0 0 0-1.437 1.437l-1.582 6.135a.5.5 0 0 1-.963 0z" />
    </svg>
  ),
  settings: (
    <svg width="1em" height="1em" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z" />
      <circle cx="12" cy="12" r="3" />
    </svg>
  ),
  message: (
    <svg width="1em" height="1em" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
    </svg>
  ),
  close: (
    <svg width="1em" height="1em" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <line x1="18" y1="6" x2="6" y2="18" />
      <line x1="6" y1="6" x2="18" y2="18" />
    </svg>
  ),
  send: (
    <svg width="1em" height="1em" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="m22 2-7 20-4-9-9-4Z" />
      <path d="M22 2 11 13" />
    </svg>
  ),
  attach: (
    <svg width="1em" height="1em" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="m21.44 11.05-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.82-2.82l8.48-8.48" />
    </svg>
  ),
  trash: (
    <svg width="1em" height="1em" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M3 6h18" />
      <path d="M8 6V4h8v2" />
      <path d="m19 6-1 14H6L5 6" />
      <path d="M10 11v6" />
      <path d="M14 11v6" />
    </svg>
  ),
};

function App() {
  const [stream, setStream] = useState(null);
  const [uploadedFile, setUploadedFile] = useState(null);
  const [dragOver, setDragOver] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [previewError, setPreviewError] = useState('');
  const [analysisError, setAnalysisError] = useState('');
  const [prediction, setPrediction] = useState('');
  const [rawPrediction, setRawPrediction] = useState('');
  const [predictions, setPredictions] = useState([]);
  const [explanation, setExplanation] = useState('');
  const [confidence, setConfidence] = useState(0);
  const [confidenceThreshold, setConfidenceThreshold] = useState(25);
  const [top1MinConfidence, setTop1MinConfidence] = useState(52);
  const [topGap, setTopGap] = useState(0);
  const [gapThreshold, setGapThreshold] = useState(12);
  const [isUncertain, setIsUncertain] = useState(false);
  const [backendStatus, setBackendStatus] = useState('checking');
  const [inferenceMode, setInferenceMode] = useState('unknown');
  const [deviceType, setDeviceType] = useState('-');
  const [modelName, setModelName] = useState('SlowFast R50 (Kinetics-400)');
  const [geminiConfigured, setGeminiConfigured] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [geminiKey, setGeminiKey] = useState('');
  const [maxUploadMb, setMaxUploadMb] = useState(200);
  const [videoValidation, setVideoValidation] = useState(null);
  const [preprocessingInfo, setPreprocessingInfo] = useState(null);
  const [webcamStatus, setWebcamStatus] = useState({ supported: true, message: '' });
  const [bufferCount, setBufferCount] = useState(0);
  const [bufferThreshold, setBufferThreshold] = useState(0);
  const [activeWorkspace, setActiveWorkspace] = useState('analysis');
  const [currentModelInfo, setCurrentModelInfo] = useState(null);
  const [recentAnalyses, setRecentAnalyses] = useState([]);
  const [isHistoryLoading, setIsHistoryLoading] = useState(false);
  const [activePanelTab, setActivePanelTab] = useState('analysis');
  const [analysisStage, setAnalysisStage] = useState('');
  const [evaluationSummary, setEvaluationSummary] = useState(null);
  const [evaluationMeta, setEvaluationMeta] = useState({
    manifestExists: false,
    manifestPath: '',
    manifestSamples: 0,
    manifestError: '',
  });
  const [isEvaluationLoading, setIsEvaluationLoading] = useState(false);
  const [isRunningEvaluation, setIsRunningEvaluation] = useState(false);
  const [reviewSamples, setReviewSamples] = useState([]);
  const [reviewCounts, setReviewCounts] = useState({ pending: 0, reviewed: 0, total: 0 });
  const [reviewFilter, setReviewFilter] = useState('pending');
  const [isReviewLoading, setIsReviewLoading] = useState(false);
  const [selectedReviewFilename, setSelectedReviewFilename] = useState(null);
  const [reviewLabelInput, setReviewLabelInput] = useState('');
  const [reviewNotesInput, setReviewNotesInput] = useState('');
  const [isSavingReview, setIsSavingReview] = useState(false);
  const [trainingOverview, setTrainingOverview] = useState(null);
  const [trainingRuns, setTrainingRuns] = useState([]);
  const [selectedTrainingRunId, setSelectedTrainingRunId] = useState(null);
  const [isTrainingLabLoading, setIsTrainingLabLoading] = useState(false);
  const [isStartingTraining, setIsStartingTraining] = useState(false);
  const [isPromotingTrainingRun, setIsPromotingTrainingRun] = useState(false);

  const [showChat, setShowChat] = useState(false);
  const [chatSessions, setChatSessions] = useState([]);
  const [activeSessionId, setActiveSessionId] = useState(null);
  const [chatMessages, setChatMessages] = useState([]);
  const [chatInput, setChatInput] = useState('');
  const [chatUpload, setChatUpload] = useState(null);
  const [chatError, setChatError] = useState('');
  const [isChatLoading, setIsChatLoading] = useState(false);
  const [isSendingChat, setIsSendingChat] = useState(false);

  const videoRef = useRef(null);
  const fileInputRef = useRef(null);
  const previewUrlRef = useRef(null);
  const chatFileInputRef = useRef(null);
  const chatScrollRef = useRef(null);

  useEffect(() => {
    return () => {
      if (previewUrlRef.current) {
        URL.revokeObjectURL(previewUrlRef.current);
      }
    };
  }, []);

  useEffect(() => {
    if (stream && videoRef.current) {
      videoRef.current.srcObject = stream;
    }
  }, [stream]);

  useEffect(() => {
    if (chatScrollRef.current) {
      chatScrollRef.current.scrollTop = chatScrollRef.current.scrollHeight;
    }
  }, [chatMessages, showChat]);

  useEffect(() => {
    if (typeof navigator === 'undefined') {
      setWebcamStatus({ supported: false, message: 'This environment does not expose browser camera APIs.' });
      return;
    }

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setWebcamStatus({
        supported: false,
        message: 'Camera access is unavailable here. Open the app through localhost or a secure browser context.',
      });
      return;
    }

    if (typeof MediaRecorder === 'undefined') {
      setWebcamStatus({
        supported: false,
        message: 'This browser cannot record webcam clips. Upload a video file instead.',
      });
      return;
    }

    setWebcamStatus({ supported: true, message: '' });
  }, []);

  useEffect(() => {
    const checkBackend = async () => {
      try {
        const response = await fetch(`${API_BASE}/health`);
        if (!response.ok) {
          setBackendStatus('offline');
          return;
        }
        const data = await response.json();
        setBackendStatus('online');
        setInferenceMode(data.inference_mode || (data.model_loaded ? 'live' : 'mock'));
        setDeviceType(data.device || '-');
        setModelName(data.model_name || 'SlowFast R50 (Kinetics-400)');
        setCurrentModelInfo(data.current_model || null);
        setGeminiConfigured(Boolean(data.gemini_configured));
        setConfidenceThreshold(Number(data.confidence_threshold || 25));
        setTop1MinConfidence(Number(data.top1_min_confidence || 52));
        setGapThreshold(Number(data.top1_top2_gap_threshold || 12));
        setMaxUploadMb(Number(data.max_upload_mb || 200));
        setBufferCount(Number(data.buffer_count || 0));
        setBufferThreshold(Number(data.buffer_threshold || 0));
        setEvaluationMeta((current) => ({
          ...current,
          manifestSamples: Number(data.evaluation_manifest_samples || 0),
        }));
      } catch {
        setBackendStatus('offline');
      }
    };

    checkBackend();
    const interval = setInterval(checkBackend, 30000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    fetchAnalysisHistory();
    fetchEvaluationSummary();
    fetchReviewSamples('pending');
    fetchTrainingLab();
  }, []);

  useEffect(() => {
    if (activePanelTab === 'review') {
      fetchReviewSamples(reviewFilter);
    }
  }, [activePanelTab, reviewFilter]);

  useEffect(() => {
    if (activeWorkspace !== 'training') return undefined;
    fetchTrainingLab();
    const interval = setInterval(fetchTrainingLab, 4000);
    return () => clearInterval(interval);
  }, [activeWorkspace]);

  const resetAnalysis = () => {
    setPrediction('');
    setRawPrediction('');
    setPredictions([]);
    setExplanation('');
    setConfidence(0);
    setTopGap(0);
    setIsUncertain(false);
    setAnalysisError('');
    setPreviewError('');
    setVideoValidation(null);
    setPreprocessingInfo(null);
  };

  const clearSelectedSource = () => {
    stopWebcam();
    if (previewUrlRef.current) {
      URL.revokeObjectURL(previewUrlRef.current);
      previewUrlRef.current = null;
    }
    setUploadedFile(null);
    if (videoRef.current) {
      videoRef.current.srcObject = null;
      videoRef.current.removeAttribute('src');
      videoRef.current.load();
    }
    resetAnalysis();
  };

  const stopWebcam = () => {
    if (!stream) return;
    stream.getTracks().forEach((track) => track.stop());
    setStream(null);
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
  };

  const startWebcam = async () => {
    if (!webcamStatus.supported) {
      setAnalysisError(webcamStatus.message || 'Webcam capture is not available in this browser.');
      return;
    }

    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({ video: true });
      if (previewUrlRef.current) {
        URL.revokeObjectURL(previewUrlRef.current);
        previewUrlRef.current = null;
      }
      setUploadedFile(null);
      resetAnalysis();
      setAnalysisError('');
      setStream(mediaStream);
      if (videoRef.current) {
        videoRef.current.removeAttribute('src');
      }
    } catch (error) {
      setAnalysisError(getWebcamErrorMessage(error));
    }
  };

  const handleFile = (file) => {
    if (!file) return;
    if (!file.type.startsWith('video/')) {
      setAnalysisError('Only video files are supported.');
      return;
    }
    if (file.size > maxUploadMb * 1024 * 1024) {
      setAnalysisError(`This file is larger than the ${maxUploadMb} MB limit.`);
      return;
    }
    if (stream) {
      stream.getTracks().forEach((track) => track.stop());
      setStream(null);
    }
    if (previewUrlRef.current) {
      URL.revokeObjectURL(previewUrlRef.current);
    }

    const previewUrl = URL.createObjectURL(file);
    previewUrlRef.current = previewUrl;
    setUploadedFile(file);
    if (videoRef.current) {
      videoRef.current.srcObject = null;
      videoRef.current.src = previewUrl;
      videoRef.current.load();
    }
    resetAnalysis();
  };

  const handleDrop = (event) => {
    event.preventDefault();
    setDragOver(false);
    handleFile(event.dataTransfer.files[0]);
  };

  const saveGeminiKey = async () => {
    const trimmed = geminiKey.trim();
    if (!trimmed) return;
    try {
      const response = await fetch(`${API_BASE}/configure`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ gemini_api_key: trimmed }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || 'Failed to save Gemini key.');
      setGeminiConfigured(Boolean(data.gemini_configured));
      setGeminiKey('');
      setShowSettings(false);
      if (showChat) {
        fetchChatHistory();
      }
    } catch (error) {
      setChatError(error.message || 'Failed to save Gemini key.');
    }
  };

  const analyzeVideo = async () => {
    setIsAnalyzing(true);
    setAnalysisStage('Preparing clip...');
    resetAnalysis();
    let fileToSend = uploadedFile;

    if (stream && !uploadedFile) {
      try {
        setAnalysisStage('Recording a short webcam clip...');
        fileToSend = await captureWebcamClip(stream, 3000);
      } catch (error) {
        setAnalysisError(error.message || 'Webcam recording failed.');
        setIsAnalyzing(false);
        setAnalysisStage('');
        return;
      }
    }

    if (!fileToSend) {
      setAnalysisError('Choose a video or start the webcam first.');
      setIsAnalyzing(false);
      setAnalysisStage('');
      return;
    }

    try {
      setAnalysisStage('Uploading video to the local backend...');
      const formData = new FormData();
      formData.append('video', fileToSend, fileToSend.name || 'webcam_clip.webm');

      const response = await fetch(`${API_BASE}/upload-video`, {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || 'Prediction failed.');

      setAnalysisStage('Reviewing the model result...');
      setPrediction(data.top_prediction);
      setRawPrediction(data.raw_top_prediction || data.top_prediction);
      setPredictions(data.predictions || []);
      setConfidence(Number(data.top_confidence || 0));
      setTopGap(Number(data.top_gap || 0));
      setExplanation('');
      setIsUncertain(Boolean(data.is_uncertain));
      setInferenceMode(data.mode || 'unknown');
      setDeviceType(data.device || '-');
      setModelName(data.model_name || modelName);
      setVideoValidation(data.video_validation || null);
      setPreprocessingInfo(data.preprocessing || null);

      if (Boolean(data.is_uncertain)) {
        setExplanation('The model found the clip readable, but the best class was not strong enough to present as a reliable answer.');
        return;
      }

      setAnalysisStage('Requesting an explanation...');
      const explainResponse = await fetch(
        `${API_BASE}/explanation?prediction=${encodeURIComponent(data.top_prediction)}&analysis_id=${encodeURIComponent(data.analysis_id)}`
      );
      const explainData = await explainResponse.json();
      if (explainResponse.ok) {
        setExplanation(explainData.explanation);
      } else {
        setExplanation('Explanation unavailable.');
      }
    } catch (error) {
      setAnalysisError(error.message || 'Unknown analysis error.');
    } finally {
      setIsAnalyzing(false);
      setAnalysisStage('');
      fetchAnalysisHistory();
    }
  };

  const fetchAnalysisHistory = async () => {
    setIsHistoryLoading(true);
    try {
      const response = await fetch(`${API_BASE}/analysis/history?limit=10`);
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || 'Unable to load recent analyses.');
      setRecentAnalyses(data.analyses || []);
    } catch (error) {
      setAnalysisError((current) => current || error.message || 'Unable to load recent analyses.');
    } finally {
      setIsHistoryLoading(false);
    }
  };

  const fetchEvaluationSummary = async () => {
    setIsEvaluationLoading(true);
    try {
      const response = await fetch(`${API_BASE}/evaluation/latest`);
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || 'Unable to load evaluation summary.');
      setEvaluationSummary(data.latest_report?.summary || null);
      setEvaluationMeta({
        manifestExists: Boolean(data.manifest_exists),
        manifestPath: data.manifest_path || '',
        manifestSamples: Number(data.manifest_samples || 0),
        manifestError: data.manifest_error || '',
      });
    } catch (error) {
      setAnalysisError((current) => current || error.message || 'Unable to load evaluation summary.');
    } finally {
      setIsEvaluationLoading(false);
    }
  };

  const runEvaluation = async () => {
    setIsRunningEvaluation(true);
    try {
      const response = await fetch(`${API_BASE}/evaluation/run`, { method: 'POST' });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || 'Unable to run evaluation.');
      setEvaluationSummary(data.summary || null);
      await fetchEvaluationSummary();
    } catch (error) {
      setAnalysisError(error.message || 'Unable to run evaluation.');
    } finally {
      setIsRunningEvaluation(false);
    }
  };

  const fetchTrainingLab = async () => {
    setIsTrainingLabLoading(true);
    try {
      const [overviewResponse, runsResponse] = await Promise.all([
        fetch(`${API_BASE}/training-lab/overview`),
        fetch(`${API_BASE}/training-runs?limit=24`),
      ]);
      const overviewData = await overviewResponse.json();
      const runsData = await runsResponse.json();
      if (!overviewResponse.ok) throw new Error(overviewData.detail || 'Unable to load Training Lab overview.');
      if (!runsResponse.ok) throw new Error(runsData.detail || 'Unable to load training runs.');

      setTrainingOverview(overviewData.overview || null);
      setTrainingRuns(runsData.runs || []);

      const nextCurrentModel = overviewData.overview?.current_model || null;
      if (nextCurrentModel) {
        setCurrentModelInfo(nextCurrentModel);
        setModelName(nextCurrentModel.name || modelName);
      }

      const preferredRun =
        (runsData.current_run && runsData.current_run.id) ||
        selectedTrainingRunId ||
        runsData.runs?.[0]?.id ||
        null;
      setSelectedTrainingRunId(preferredRun);
    } catch (error) {
      setAnalysisError((current) => current || error.message || 'Unable to load Training Lab.');
    } finally {
      setIsTrainingLabLoading(false);
    }
  };

  const startTrainingRun = async () => {
    setIsStartingTraining(true);
    setAnalysisError('');
    try {
      const response = await fetch(`${API_BASE}/training-runs/start`, { method: 'POST' });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || 'Unable to start pilot training.');
      setSelectedTrainingRunId(data.run?.id || null);
      await fetchTrainingLab();
    } catch (error) {
      setAnalysisError(error.message || 'Unable to start pilot training.');
    } finally {
      setIsStartingTraining(false);
    }
  };

  const promoteTrainingRun = async (runId) => {
    if (!runId) return;
    setIsPromotingTrainingRun(true);
    setAnalysisError('');
    try {
      const response = await fetch(`${API_BASE}/training-runs/${encodeURIComponent(runId)}/promote`, { method: 'POST' });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || 'Unable to promote training run.');
      setCurrentModelInfo(data.current_model || null);
      setModelName(data.current_model?.name || modelName);
      await fetchTrainingLab();
    } catch (error) {
      setAnalysisError(error.message || 'Unable to promote training run.');
    } finally {
      setIsPromotingTrainingRun(false);
    }
  };

  const resetToDefaultModel = async () => {
    setAnalysisError('');
    try {
      const response = await fetch(`${API_BASE}/models/current/reset`, { method: 'POST' });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || 'Unable to reset the active model.');
      setCurrentModelInfo(data.current_model || null);
      setModelName(data.current_model?.name || 'SlowFast R50 (Kinetics-400)');
      await fetchTrainingLab();
    } catch (error) {
      setAnalysisError(error.message || 'Unable to reset the active model.');
    }
  };

  const fetchReviewSamples = async (status = reviewFilter) => {
    setIsReviewLoading(true);
    try {
      const response = await fetch(`${API_BASE}/review/samples?limit=24&status=${encodeURIComponent(status)}`);
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || 'Unable to load review samples.');
      const samples = data.samples || [];
      setReviewSamples(samples);
      setReviewCounts(data.counts || { pending: 0, reviewed: 0, total: 0 });

      const stillSelected = samples.find((item) => item.filename === selectedReviewFilename);
      const nextSelected = stillSelected || samples[0] || null;
      setSelectedReviewFilename(nextSelected?.filename || null);
      setReviewLabelInput(nextSelected?.reviewed_label || nextSelected?.prediction || '');
      setReviewNotesInput(nextSelected?.review_notes || '');
    } catch (error) {
      setAnalysisError((current) => current || error.message || 'Unable to load review samples.');
    } finally {
      setIsReviewLoading(false);
    }
  };

  const selectReviewSample = (sample) => {
    setSelectedReviewFilename(sample?.filename || null);
    setReviewLabelInput(sample?.reviewed_label || sample?.prediction || '');
    setReviewNotesInput(sample?.review_notes || '');
  };

  const saveReviewedLabel = async () => {
    if (!selectedReviewFilename || !reviewLabelInput.trim()) return;
    setIsSavingReview(true);
    try {
      const response = await fetch(`${API_BASE}/review/samples/${encodeURIComponent(selectedReviewFilename)}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          label: reviewLabelInput.trim(),
          notes: reviewNotesInput.trim(),
        }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || 'Unable to save reviewed label.');
      await fetchReviewSamples(reviewFilter);
      await fetchEvaluationSummary();
    } catch (error) {
      setAnalysisError(error.message || 'Unable to save reviewed label.');
    } finally {
      setIsSavingReview(false);
    }
  };

  const deleteAnalysisItem = async (analysisId) => {
    try {
      const response = await fetch(`${API_BASE}/analysis/history/${analysisId}`, {
        method: 'DELETE',
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || 'Unable to delete analysis.');
      setRecentAnalyses((current) => current.filter((item) => item.analysis_id !== analysisId));
    } catch (error) {
      setAnalysisError(error.message || 'Unable to delete analysis.');
    }
  };

  const downloadFile = async (url, filename, errorMessage) => {
    try {
      const response = await fetch(url);
      if (!response.ok) {
        const data = await response.json().catch(() => ({}));
        throw new Error(data.detail || errorMessage);
      }
      const blob = await response.blob();
      const objectUrl = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = objectUrl;
      link.download = filename;
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(objectUrl);
    } catch (error) {
      setAnalysisError(error.message || errorMessage);
    }
  };

  const downloadAnalysisHistory = async (format) => {
    await downloadFile(
      `${API_BASE}/analysis/history/export?format=${encodeURIComponent(format)}`,
      `analysis_history.${format}`,
      'Unable to export analysis history.'
    );
  };

  const downloadLatestEvaluationReport = async () => {
    await downloadFile(
      `${API_BASE}/evaluation/latest/export`,
      'latest_evaluation_report.json',
      'Unable to export the latest evaluation report.'
    );
  };

  const downloadReviewedManifest = async () => {
    await downloadFile(
      `${API_BASE}/datasets/reviewed/manifest/export`,
      'reviewed_manifest.jsonl',
      'Unable to export the reviewed dataset manifest.'
    );
  };

  const downloadActiveChatSession = async () => {
    if (!activeSessionId) return;
    await downloadFile(
      `${API_BASE}/chat/history/${activeSessionId}/export`,
      `chat_${activeSessionId}.md`,
      'Unable to export this chat session.'
    );
  };

  const seedChatPrompt = async (prompt) => {
    if (!showChat) {
      await openChat();
    }
    setChatInput(prompt);
  };

  const fetchChatHistory = async () => {
    setIsChatLoading(true);
    try {
      const response = await fetch(`${API_BASE}/chat/history`);
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || 'Unable to load chat history.');
      const sessions = data.sessions || [];
      setChatSessions(sessions);
      setGeminiConfigured(Boolean(data.gemini_configured));
      if (!activeSessionId && sessions.length > 0) {
        await loadChatSession(sessions[0].id);
      } else if (activeSessionId && !sessions.some((session) => session.id === activeSessionId)) {
        setActiveSessionId(null);
        setChatMessages([]);
      }
    } catch (error) {
      setChatError(error.message || 'Unable to load chat history.');
    } finally {
      setIsChatLoading(false);
    }
  };

  const loadChatSession = async (sessionId) => {
    try {
      const response = await fetch(`${API_BASE}/chat/history/${sessionId}`);
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || 'Unable to load chat.');
      setActiveSessionId(sessionId);
      setChatMessages(data.session.messages || []);
      setChatError('');
    } catch (error) {
      setChatError(error.message || 'Unable to load chat.');
    }
  };

  const deleteChatSession = async (sessionId) => {
    try {
      const response = await fetch(`${API_BASE}/chat/history/${sessionId}`, {
        method: 'DELETE',
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || 'Unable to delete chat.');

      const remaining = chatSessions.filter((session) => session.id !== sessionId);
      setChatSessions(remaining);
      if (activeSessionId === sessionId) {
        setActiveSessionId(null);
        setChatMessages([]);
        if (remaining.length > 0) {
          loadChatSession(remaining[0].id);
        }
      }
    } catch (error) {
      setChatError(error.message || 'Unable to delete chat.');
    }
  };

  const openChat = async () => {
    setShowChat(true);
    await fetchChatHistory();
  };

  const renameActiveChatSession = async () => {
    if (!activeSessionId) return;
    const session = chatSessions.find((item) => item.id === activeSessionId);
    const nextTitle = window.prompt('Rename this conversation:', session?.title || 'New chat');
    if (!nextTitle || !nextTitle.trim()) return;

    try {
      const response = await fetch(`${API_BASE}/chat/history/${activeSessionId}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title: nextTitle.trim() }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || 'Unable to rename chat.');
      await fetchChatHistory();
    } catch (error) {
      setChatError(error.message || 'Unable to rename chat.');
    }
  };

  const startNewChat = () => {
    setActiveSessionId(null);
    setChatMessages([]);
    setChatInput('');
    setChatUpload(null);
    setChatError('');
  };

  const sendChatMessage = async () => {
    const trimmed = chatInput.trim();
    if (!trimmed && !chatUpload) return;

    setIsSendingChat(true);
    setChatError('');
    try {
      const formData = new FormData();
      if (trimmed) formData.append('message', trimmed);
      if (activeSessionId) formData.append('session_id', activeSessionId);
      if (chatUpload) formData.append('video', chatUpload, chatUpload.name);

      const response = await fetch(`${API_BASE}/chat/send`, {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || 'Unable to send message.');

      setActiveSessionId(data.session.id);
      setChatMessages((current) => {
        const base = current.length && activeSessionId === data.session.id ? current : [];
        return [...base, ...(data.messages || [])];
      });
      setChatInput('');
      setChatUpload(null);
      if (chatFileInputRef.current) {
        chatFileInputRef.current.value = '';
      }
      await fetchChatHistory();
    } catch (error) {
      setChatError(error.message || 'Unable to send message.');
    } finally {
      setIsSendingChat(false);
    }
  };

  const analysisHeading = isUncertain ? 'Uncertain result' : prediction ? formatLabel(prediction) : 'No analysis yet';
  const analysisSubheading = isUncertain
    ? `Held back because confidence or separation was too weak. Raw candidate: ${formatLabel(rawPrediction)}.`
    : prediction
      ? `Top confidence ${confidence.toFixed(1)}%, gap to next class ${topGap.toFixed(1)}%.`
      : 'Upload a short clip or capture from webcam to analyze.';
  const selectedReviewSample = reviewSamples.find((item) => item.filename === selectedReviewFilename) || null;
  const selectedTrainingRun = trainingRuns.find((item) => item.id === selectedTrainingRunId) || trainingOverview?.current_run || null;
  const workspaceTitle = activeWorkspace === 'training' ? 'Training Lab' : 'NeuralVision AI';
  const workspaceSubtitle = activeWorkspace === 'training'
    ? 'Launch the pilot trainer, monitor progress, and promote checkpoints when they are ready.'
    : 'Local video analysis with a simpler workspace and built-in chat.';

  return (
    <div className="app-wrapper">
      <div className="app-container">
        <nav className="sidebar">
          <div className="sidebar-logo">N</div>
          <button
            className={`sidebar-btn ${activeWorkspace === 'analysis' ? 'active' : ''}`}
            title="Analysis"
            onClick={() => setActiveWorkspace('analysis')}
          >
            {Icons.video}
          </button>
          <button
            className={`sidebar-btn ${activeWorkspace === 'training' ? 'active' : ''}`}
            title="Training Lab"
            onClick={() => setActiveWorkspace('training')}
          >
            {Icons.sparkles}
          </button>
          <button className="sidebar-btn" title="Send Messages" onClick={openChat}>{Icons.message}</button>
          <div className="sidebar-spacer" />
          <button className="sidebar-btn" title="Settings" onClick={() => setShowSettings(true)}>{Icons.settings}</button>
        </nav>

        <header className="topbar simple-topbar">
          <div className="topbar-copy">
            <div className="topbar-title">{workspaceTitle}</div>
            <div className="topbar-subtitle">{workspaceSubtitle}</div>
          </div>
          <div className="topbar-actions">
            <div className="mode-switch">
              <button className={`mode-switch-btn ${activeWorkspace === 'analysis' ? 'active' : ''}`} onClick={() => setActiveWorkspace('analysis')}>
                Analysis
              </button>
              <button className={`mode-switch-btn ${activeWorkspace === 'training' ? 'active' : ''}`} onClick={() => setActiveWorkspace('training')}>
                Training Lab
              </button>
            </div>
            <div className={`status-pill ${backendStatus === 'online' ? 'online' : 'offline'}`}>
              {backendStatus === 'online' ? 'Backend Ready' : backendStatus === 'checking' ? 'Checking' : 'Backend Offline'}
            </div>
            <button className="btn btn-primary compact-btn" onClick={openChat}>
              {Icons.message} Send Messages
            </button>
          </div>
        </header>

        <main className="main-content">
          {activeWorkspace === 'analysis' ? (
            <div className="simplified-layout">
          <section className="glass-panel source-panel">
            <div className="section-header">
              <div>
                <div className="section-title">Video source</div>
                <div className="section-caption">Upload a clip or capture 3 seconds from your webcam.</div>
              </div>
              <div className="section-tags">
                {stream && <span className="panel-badge live">Live</span>}
                {uploadedFile && <span className="panel-badge ready">Loaded</span>}
              </div>
            </div>

            <div className="video-viewport large-viewport">
              {(stream || uploadedFile) ? (
                <>
                  <video
                    ref={videoRef}
                    autoPlay
                    playsInline
                    muted
                    controls={!stream}
                    onLoadedMetadata={() => setPreviewError('')}
                    onError={() => setPreviewError('This video cannot be previewed in the browser. Try MP4/H.264 or WEBM.')}
                  />
                  {stream && (
                    <div className="video-overlay">
                      <div className="rec-indicator"><span className="dot" /> LIVE</div>
                      <span>Webcam feed</span>
                    </div>
                  )}
                  {!stream && uploadedFile && !previewError && (
                    <div className="video-overlay">
                      <span>{uploadedFile.name}</span>
                      <span>{(uploadedFile.size / 1e6).toFixed(1)} MB</span>
                    </div>
                  )}
                  {previewError && (
                    <div className="video-preview-error">
                      <strong>Preview unavailable</strong>
                      <span>{previewError}</span>
                    </div>
                  )}
                </>
              ) : (
                <div className="video-empty">
                  <span className="icon">{Icons.video}</span>
                  <p>No video selected yet</p>
                </div>
              )}
            </div>

            <div className="source-controls">
              <div className="controls-row">
                <button
                  className={`btn btn-full ${stream ? 'btn-danger' : ''}`}
                  onClick={stream ? stopWebcam : startWebcam}
                  disabled={!stream && !webcamStatus.supported}
                >
                  {stream ? Icons.stop : Icons.camera}
                  {stream ? 'Stop Webcam' : 'Start Webcam'}
                </button>
                <button className="btn btn-full" onClick={() => fileInputRef.current.click()}>
                  {Icons.upload} Upload Video
                </button>
                <button className="btn btn-full" onClick={clearSelectedSource} disabled={!stream && !uploadedFile}>
                  {Icons.close} Clear Source
                </button>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="video/*"
                  style={{ display: 'none' }}
                  onChange={(event) => {
                    handleFile(event.target.files[0]);
                    event.target.value = '';
                  }}
                />
              </div>

              <div
                className={`drop-zone ${dragOver ? 'drag-over' : ''}`}
                onClick={() => fileInputRef.current.click()}
                onDragOver={(event) => { event.preventDefault(); setDragOver(true); }}
                onDragLeave={() => setDragOver(false)}
                onDrop={handleDrop}
              >
                <div className="icon">{Icons.upload}</div>
                <p>Drag and drop a video here, or <span className="highlight">browse</span></p>
                <p className="drop-note">Supported: MP4, AVI, MOV, WEBM. Max {maxUploadMb} MB.</p>
              </div>

              <button className="btn btn-primary analyze-btn" onClick={analyzeVideo} disabled={isAnalyzing || backendStatus !== 'online' || (!stream && !uploadedFile)}>
                {isAnalyzing ? 'Analyzing...' : 'Analyze Video'}
              </button>

              {!webcamStatus.supported && <div className="inline-note">{webcamStatus.message}</div>}
              {analysisError && <div className="inline-error">{analysisError}</div>}
            </div>
          </section>

          <section className="analysis-column">
            <div className="glass-panel analysis-panel">
              <div className="section-header">
                <div>
                  <div className="section-title">Analysis</div>
                  <div className="section-caption">More conservative output for a general pretrained action model.</div>
                </div>
                <div className="meta-chip">{formatDevice(deviceType)} / {inferenceMode === 'live' ? 'Live' : 'Mock'}</div>
              </div>

              {isAnalyzing ? (
                <div className="spinner-wrap compact-spinner">
                  <div className="spinner" />
                  <div className="spinner-text">{analysisStage || 'Running SlowFast inference...'}</div>
                </div>
              ) : (
                <div className="analysis-body">
                  <div className="analysis-summary">
                    <div className="analysis-title-row">
                      <h2>{analysisHeading}</h2>
                      {prediction && (
                        <span className={`analysis-status ${isUncertain ? 'warn' : 'ok'}`}>
                          {isUncertain ? 'Held back' : 'Accepted'}
                        </span>
                      )}
                    </div>
                    <p>{analysisSubheading}</p>
                  </div>

                  <div className="metric-strip">
                    <div className="metric-item">
                      <span>Top confidence <em className="metric-help" title="The confidence score for the highest-ranked class before uncertainty logic is applied.">?</em></span>
                      <strong>{confidence ? `${confidence.toFixed(1)}%` : '-'}</strong>
                    </div>
                    <div className="metric-item">
                      <span>Minimum confidence <em className="metric-help" title="Predictions below this threshold are held back as uncertain even if they are the top class.">?</em></span>
                      <strong>{top1MinConfidence.toFixed(1)}%</strong>
                    </div>
                    <div className="metric-item">
                      <span>Gap to next class <em className="metric-help" title="A small gap means the model is not clearly separating the top class from the runner-up.">?</em></span>
                      <strong>{topGap ? `${topGap.toFixed(1)}%` : '-'}</strong>
                    </div>
                    <div className="metric-item">
                      <span>Gap threshold <em className="metric-help" title="If the gap to the next class is below this threshold, the app reports the clip as uncertain.">?</em></span>
                      <strong>{gapThreshold.toFixed(1)}%</strong>
                    </div>
                  </div>

                  {predictions.length > 0 && (
                    <div className="prediction-list">
                      {predictions.slice(0, 5).map((item, index) => (
                        <div key={`${item.label}-${index}`} className="prediction-row">
                          <span className="prediction-rank">{index + 1}</span>
                          <span className="prediction-label">{formatLabel(item.label)}</span>
                          <span className="prediction-confidence">{Number(item.confidence || 0).toFixed(1)}%</span>
                        </div>
                      ))}
                    </div>
                  )}

                  {explanation && (
                    <div className="clean-card">
                      <div className="clean-card-label">Explanation</div>
                      <p>{explanation}</p>
                    </div>
                  )}

                  {videoValidation && (
                    <div className="clean-card muted-card">
                      <div className="clean-card-label">Video check</div>
                      <p>
                        {videoValidation.readable_frames} readable frames, {videoValidation.width}x{videoValidation.height}, average frame std {videoValidation.average_frame_std}.
                        {preprocessingInfo?.selected_frame_count ? ` Selected ${preprocessingInfo.selected_frame_count} frames after trimming.` : ''}
                        {preprocessingInfo?.normalized_container ? ' The clip was normalized to MP4 for inference stability.' : ''}
                      </p>
                    </div>
                  )}

                  <div className="action-chip-row">
                    <button className="btn" onClick={() => seedChatPrompt('Why was the last clip uncertain?')}>
                      Ask why uncertain
                    </button>
                    <button className="btn" onClick={() => seedChatPrompt('Summarize the latest analysis in simple words.')}>
                      Send to chat
                    </button>
                  </div>
                </div>
              )}
            </div>

            <div className="panel-tabs">
              {['analysis', 'review', 'history', 'evaluation'].map((tab) => (
                <button
                  key={tab}
                  className={`panel-tab ${activePanelTab === tab ? 'active' : ''}`}
                  onClick={() => setActivePanelTab(tab)}
                >
                  {tab === 'analysis' ? 'Analysis' : tab === 'review' ? 'Review' : tab === 'history' ? 'History' : 'Evaluation'}
                </button>
              ))}
            </div>

            {activePanelTab === 'analysis' && (
              <div className="glass-panel helper-panel">
                <div className="section-header">
                  <div>
                    <div className="section-title">System status</div>
                    <div className="section-caption">Quick visibility into the local backend, model path, review buffer, and exports.</div>
                  </div>
                </div>
                <div className="system-grid">
                  <div className="system-card">
                    <span>Model</span>
                    <strong>{modelName}</strong>
                  </div>
                  <div className="system-card">
                    <span>Compute</span>
                    <strong>{formatDevice(deviceType)}</strong>
                  </div>
                  <div className="system-card">
                    <span>Mode</span>
                    <strong>{inferenceMode === 'live' ? 'Live inference' : 'Mock fallback'}</strong>
                  </div>
                  <div className="system-card">
                    <span>Review buffer</span>
                    <strong>{bufferThreshold ? `${bufferCount}/${bufferThreshold}` : bufferCount}</strong>
                  </div>
                </div>
                <div className="action-chip-row">
                  <button className="btn" onClick={() => downloadAnalysisHistory('json')}>Export JSON</button>
                  <button className="btn" onClick={() => downloadAnalysisHistory('csv')}>Export CSV</button>
                  <button className="btn" onClick={downloadReviewedManifest}>Reviewed Manifest</button>
                  <button className="btn btn-primary" onClick={openChat}>{Icons.message} Open Chat Workspace</button>
                </div>
              </div>
            )}

            {activePanelTab === 'evaluation' && (
              <div className="glass-panel helper-panel">
                <div className="section-header">
                  <div>
                    <div className="section-title">Evaluation</div>
                    <div className="section-caption">Measure the current model on your own labeled clips before trusting it more deeply.</div>
                  </div>
                  <button className="btn" onClick={fetchEvaluationSummary}>Refresh</button>
                </div>

                <div className="evaluation-summary">
                  <div className="system-grid">
                    <div className="system-card">
                      <span>Manifest samples</span>
                      <strong>{evaluationMeta.manifestExists ? evaluationMeta.manifestSamples : 'Missing'}</strong>
                    </div>
                    <div className="system-card">
                      <span>Accepted accuracy</span>
                      <strong>{evaluationSummary ? `${Number(evaluationSummary.accepted_accuracy || 0).toFixed(1)}%` : '-'}</strong>
                    </div>
                    <div className="system-card">
                      <span>Raw top-1 accuracy</span>
                      <strong>{evaluationSummary ? `${Number(evaluationSummary.raw_top1_accuracy || 0).toFixed(1)}%` : '-'}</strong>
                    </div>
                    <div className="system-card">
                      <span>Uncertainty rate</span>
                      <strong>{evaluationSummary ? `${Number(evaluationSummary.uncertainty_rate || 0).toFixed(1)}%` : '-'}</strong>
                    </div>
                  </div>

                  {evaluationMeta.manifestError && (
                    <div className="inline-error">{evaluationMeta.manifestError}</div>
                  )}

                  {!evaluationMeta.manifestExists && !evaluationMeta.manifestError && (
                    <div className="clean-card muted-card">
                      <div className="clean-card-label">Manifest needed</div>
                      <p>Create <code>data/evaluation/manifest.jsonl</code> with one JSON line per labeled clip before running evaluation.</p>
                    </div>
                  )}

                  {evaluationMeta.manifestExists && !evaluationSummary && !isEvaluationLoading && (
                    <div className="clean-card muted-card">
                      <div className="clean-card-label">Ready to run</div>
                      <p>The manifest is present. Run evaluation to generate the first report for this model and threshold setup.</p>
                    </div>
                  )}

                  {isEvaluationLoading && (
                    <div className="history-note">Loading evaluation summary...</div>
                  )}

                  {evaluationSummary && (
                    <div className="evaluation-meta">
                      <span>Evaluated {evaluationSummary.evaluated_samples} / {evaluationSummary.total_samples} clips</span>
                      <span>Accepted {evaluationSummary.accepted_predictions}</span>
                      <span>Uncertain {evaluationSummary.uncertain_predictions}</span>
                    </div>
                  )}

                  <div className="action-chip-row">
                    <button
                      className="btn btn-primary launcher-btn"
                      onClick={runEvaluation}
                      disabled={isRunningEvaluation || !evaluationMeta.manifestExists || backendStatus !== 'online'}
                    >
                      {isRunningEvaluation ? 'Running evaluation...' : 'Run Evaluation'}
                    </button>
                    <button className="btn" onClick={downloadLatestEvaluationReport} disabled={!evaluationSummary}>
                      Download report
                    </button>
                  </div>
                </div>
              </div>
            )}

            {activePanelTab === 'history' && (
              <div className="glass-panel helper-panel">
                <div className="section-header">
                  <div>
                    <div className="section-title">Recent analyses</div>
                    <div className="section-caption">Your last runs stay visible here so the app feels stateful and easier to review.</div>
                  </div>
                  <div className="section-tags">
                    <button className="btn" onClick={fetchAnalysisHistory}>Refresh</button>
                    <button className="btn" onClick={() => downloadAnalysisHistory('csv')}>CSV</button>
                  </div>
                </div>

                <div className="analysis-history-list">
                  {isHistoryLoading && <div className="history-note">Loading recent analyses...</div>}
                  {!isHistoryLoading && recentAnalyses.length === 0 && (
                    <div className="history-note">No analysis history yet. Run a clip to populate this panel.</div>
                  )}
                  {recentAnalyses.map((item) => (
                    <div key={item.analysis_id} className="analysis-history-item">
                      {item.thumbnail_url && (
                        <img
                          className="analysis-thumb"
                          src={`${API_BASE}${item.thumbnail_url}`}
                          alt={item.filename || 'analysis thumbnail'}
                        />
                      )}
                      <div className="analysis-history-copy">
                        <div className="analysis-history-top">
                          <strong>{item.is_uncertain ? 'Uncertain' : formatLabel(item.top_prediction)}</strong>
                          <span className={`history-pill ${item.is_uncertain ? 'warn' : 'ok'}`}>
                            {item.is_uncertain ? 'Held back' : `${Number(item.top_confidence || 0).toFixed(1)}%`}
                          </span>
                        </div>
                        <div className="analysis-history-meta">
                          <span>{item.filename || 'Uploaded clip'}</span>
                          <span>{item.model_name || 'Unknown model'}</span>
                          <span>{formatTimestamp(item.timestamp)}</span>
                        </div>
                      </div>
                      <button className="chat-session-delete" title="Delete analysis" onClick={() => deleteAnalysisItem(item.analysis_id)}>
                        {Icons.trash}
                      </button>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {activePanelTab === 'review' && (
              <div className="glass-panel helper-panel">
                <div className="section-header">
                  <div>
                    <div className="section-title">Review buffer</div>
                    <div className="section-caption">Turn machine guesses into reviewed dataset entries for future training.</div>
                  </div>
                  <div className="section-tags">
                    {['pending', 'reviewed', 'all'].map((status) => (
                      <button
                        key={status}
                        className={`mini-filter ${reviewFilter === status ? 'active' : ''}`}
                        onClick={() => setReviewFilter(status)}
                      >
                        {status}
                      </button>
                    ))}
                  </div>
                </div>

                <div className="review-summary-row">
                  <div className="history-pill warn">Pending {reviewCounts.pending}</div>
                  <div className="history-pill ok">Reviewed {reviewCounts.reviewed}</div>
                  <div className="history-pill">Total {reviewCounts.total}</div>
                  <button className="btn" onClick={() => fetchReviewSamples(reviewFilter)}>Refresh</button>
                  <button className="btn" onClick={downloadReviewedManifest}>Export Manifest</button>
                </div>

                <div className="review-layout">
                  <div className="review-list">
                    {isReviewLoading && <div className="history-note">Loading buffered clips...</div>}
                    {!isReviewLoading && reviewSamples.length === 0 && (
                      <div className="history-note">No clips found for this review filter yet.</div>
                    )}
                    {reviewSamples.map((sample) => (
                      <button
                        key={sample.filename}
                        className={`review-list-item ${selectedReviewFilename === sample.filename ? 'active' : ''}`}
                        onClick={() => selectReviewSample(sample)}
                      >
                        <img
                          className="review-thumb"
                          src={`${API_BASE}${sample.thumbnail_url}`}
                          alt={sample.filename}
                        />
                        <div className="review-list-copy">
                          <strong>{sample.reviewed ? formatLabel(sample.reviewed_label || sample.prediction) : formatLabel(sample.prediction)}</strong>
                          <span>{sample.filename}</span>
                          <span>{formatTimestamp(sample.timestamp)}</span>
                        </div>
                      </button>
                    ))}
                  </div>

                  <div className="review-detail">
                    {!selectedReviewSample ? (
                      <div className="clean-card muted-card">
                        <div className="clean-card-label">No selection</div>
                        <p>Select a buffered clip to review its suggested label and save a reviewed entry.</p>
                      </div>
                    ) : (
                      <>
                        <div className="review-video-wrap">
                          <video
                            className="review-video"
                            src={`${API_BASE}${selectedReviewSample.video_url}`}
                            controls
                            preload="metadata"
                          />
                        </div>

                        <div className="clean-card">
                          <div className="clean-card-label">Suggested label</div>
                          <p>{formatLabel(selectedReviewSample.prediction || 'unknown')}</p>
                        </div>

                        <div className="review-form">
                          <label className="review-label">
                            Reviewed label
                            <input
                              className="review-input"
                              value={reviewLabelInput}
                              onChange={(event) => setReviewLabelInput(event.target.value)}
                              placeholder="Enter the correct action label"
                            />
                          </label>

                          <label className="review-label">
                            Notes
                            <textarea
                              className="review-textarea"
                              rows={4}
                              value={reviewNotesInput}
                              onChange={(event) => setReviewNotesInput(event.target.value)}
                              placeholder="Optional notes about the clip or why the model was wrong"
                            />
                          </label>

                          <div className="action-chip-row">
                            <button className="btn" onClick={() => setReviewLabelInput(selectedReviewSample.prediction || '')}>
                              Use suggestion
                            </button>
                            <button className="btn btn-primary" onClick={saveReviewedLabel} disabled={isSavingReview || !reviewLabelInput.trim()}>
                              {isSavingReview ? 'Saving review...' : 'Save reviewed label'}
                            </button>
                          </div>
                        </div>
                      </>
                    )}
                  </div>
                </div>
              </div>
            )}
          </section>
            </div>
          ) : (
            <section className="training-layout">
              <div className="training-main-column">
                <div className="glass-panel helper-panel">
                  <div className="section-header">
                    <div>
                      <div className="section-title">Training overview</div>
                      <div className="section-caption">The current analysis app stays intact while pilot training runs live in this separate workspace.</div>
                    </div>
                    <div className="section-tags">
                      <button className="btn" onClick={fetchTrainingLab}>Refresh</button>
                      <button className="btn" onClick={resetToDefaultModel}>Use Default SlowFast</button>
                    </div>
                  </div>

                  <div className="system-grid">
                    <div className="system-card">
                      <span>Active model</span>
                      <strong>{currentModelInfo?.name || modelName}</strong>
                    </div>
                    <div className="system-card">
                      <span>Dataset clips</span>
                      <strong>{trainingOverview?.dataset?.total_clips ?? '-'}</strong>
                    </div>
                    <div className="system-card">
                      <span>Last run</span>
                      <strong>{trainingOverview?.latest_run?.status || 'No runs yet'}</strong>
                    </div>
                    <div className="system-card">
                      <span>Latest checkpoint</span>
                      <strong>{trainingOverview?.latest_checkpoint ? 'Available' : 'None yet'}</strong>
                    </div>
                  </div>

                  <div className="clean-card muted-card training-note-card">
                    <div className="clean-card-label">Pilot dataset</div>
                    <p>Drop clips into <code>{trainingOverview?.dataset?.path || 'data/training_lab/basic_actions_dataset'}</code> under the five class folders: clap, wave, punch, talk, and walk.</p>
                  </div>

                  <div className="training-class-grid">
                    {(trainingOverview?.dataset?.classes || []).map((label) => (
                      <div key={label} className="history-pill">
                        {label}: {trainingOverview?.dataset?.counts?.[label] ?? 0}
                      </div>
                    ))}
                  </div>

                  <div className="action-chip-row">
                    <button
                      className="btn btn-primary launcher-btn"
                      onClick={startTrainingRun}
                      disabled={isStartingTraining || backendStatus !== 'online'}
                    >
                      {isStartingTraining ? 'Starting training...' : 'Start Basic Actions Pilot'}
                    </button>
                    <button className="btn" onClick={fetchTrainingLab}>
                      Reload status
                    </button>
                  </div>
                </div>

                <div className="glass-panel helper-panel">
                  <div className="section-header">
                    <div>
                      <div className="section-title">Live run monitor</div>
                      <div className="section-caption">Queued, running, completed, or failed states stay visible here with recent logs and metrics.</div>
                    </div>
                  </div>

                  {isTrainingLabLoading && !selectedTrainingRun ? (
                    <div className="history-note">Loading Training Lab...</div>
                  ) : !selectedTrainingRun ? (
                    <div className="history-note">No pilot run selected yet. Start one or pick a previous run from the right.</div>
                  ) : (
                    <div className="training-run-detail">
                      <div className="metric-strip">
                        <div className="metric-item">
                          <span>Status</span>
                          <strong>{selectedTrainingRun.status}</strong>
                        </div>
                        <div className="metric-item">
                          <span>Stage</span>
                          <strong>{formatLabel(selectedTrainingRun.stage || '-')}</strong>
                        </div>
                        <div className="metric-item">
                          <span>Epoch</span>
                          <strong>{selectedTrainingRun.current_epoch || 0}/{selectedTrainingRun.total_epochs || 0}</strong>
                        </div>
                        <div className="metric-item">
                          <span>Best val acc</span>
                          <strong>{selectedTrainingRun.summary?.best_val_accuracy ? `${Number(selectedTrainingRun.summary.best_val_accuracy).toFixed(1)}%` : '-'}</strong>
                        </div>
                      </div>

                      {selectedTrainingRun.summary?.test_accuracy && (
                        <div className="clean-card">
                          <div className="clean-card-label">Checkpoint summary</div>
                          <p>
                            Test accuracy {Number(selectedTrainingRun.summary.test_accuracy).toFixed(1)}% across
                            {' '}{selectedTrainingRun.summary.usable_samples} usable clips.
                          </p>
                        </div>
                      )}

                      <div className="training-log-panel">
                        {(selectedTrainingRun.recent_logs || []).length === 0 ? (
                          <div className="history-note">Logs will appear here once the run starts working through the dataset.</div>
                        ) : (
                          (selectedTrainingRun.recent_logs || []).slice().reverse().map((line, index) => (
                            <div key={`${selectedTrainingRun.id}-log-${index}`} className="training-log-line">{line}</div>
                          ))
                        )}
                      </div>

                      <div className="action-chip-row">
                        <button
                          className="btn btn-primary"
                          onClick={() => promoteTrainingRun(selectedTrainingRun.id)}
                          disabled={isPromotingTrainingRun || selectedTrainingRun.status !== 'completed'}
                        >
                          {isPromotingTrainingRun ? 'Promoting...' : 'Promote to Analysis'}
                        </button>
                      </div>
                    </div>
                  )}
                </div>
              </div>

              <div className="training-side-column">
                <div className="glass-panel helper-panel">
                  <div className="section-header">
                    <div>
                      <div className="section-title">Past runs</div>
                      <div className="section-caption">Every run keeps its own status, metrics, and checkpoint path.</div>
                    </div>
                  </div>

                  <div className="analysis-history-list">
                    {trainingRuns.length === 0 && !isTrainingLabLoading && (
                      <div className="history-note">No training runs yet. The first successful run will also create the first pilot checkpoint.</div>
                    )}
                    {trainingRuns.map((run) => (
                      <button
                        key={run.id}
                        className={`review-list-item ${selectedTrainingRunId === run.id ? 'active' : ''}`}
                        onClick={() => setSelectedTrainingRunId(run.id)}
                      >
                        <div className="review-list-copy">
                          <strong>{run.name}</strong>
                          <span>{formatLabel(run.status)}</span>
                          <span>{formatTimestamp(run.created_at)}</span>
                        </div>
                        <span className={`history-pill ${run.status === 'completed' ? 'ok' : run.status === 'failed' ? 'warn' : ''}`}>
                          {run.checkpoint_path ? 'Checkpoint' : 'No checkpoint'}
                        </span>
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            </section>
          )}
        </main>

        {showSettings && (
          <div className="modal-overlay" onClick={() => setShowSettings(false)}>
            <div className="modal-panel" onClick={(event) => event.stopPropagation()}>
              <div className="panel-header">
                <span className="panel-title"><span className="icon">{Icons.settings}</span> Settings</span>
                <button className="btn" onClick={() => setShowSettings(false)} style={{ padding: '4px 10px', fontSize: 12 }}>x</button>
              </div>
              <div style={{ padding: 16 }}>
                <div style={{ marginBottom: 16 }}>
                  <label style={{ fontSize: 12, fontWeight: 600, color: '#94a3b8', display: 'block', marginBottom: 6 }}>Gemini API Key</label>
                  <div style={{ display: 'flex', gap: 8 }}>
                    <input
                      type="password"
                      value={geminiKey}
                      onChange={(event) => setGeminiKey(event.target.value)}
                      placeholder="Enter your Gemini API key..."
                      style={{ flex: 1, padding: '8px 12px', background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8, color: '#f1f5f9', fontSize: 13, outline: 'none' }}
                    />
                    <button className="btn btn-primary" onClick={saveGeminiKey} style={{ padding: '8px 16px', fontSize: 12 }}>Save</button>
                  </div>
                  <p style={{ fontSize: 11, color: geminiConfigured ? '#22c55e' : '#94a3b8', marginTop: 6 }}>
                    {geminiConfigured ? 'Stored on the backend and ready for explanations and chat.' : 'Needed only for Gemini-powered explanations and chat replies.'}
                  </p>
                </div>
              </div>
            </div>
          </div>
        )}

        {showChat && (
          <div className="chat-overlay" onClick={() => setShowChat(false)}>
            <div className="chat-shell clean-chat-shell" onClick={(event) => event.stopPropagation()}>
              <aside className="chat-sidebar clean-chat-sidebar">
                <div className="chat-sidebar-header">
                  <div>
                    <div className="chat-sidebar-title">Chats</div>
                    <div className="chat-sidebar-subtitle">Previous conversations stay here.</div>
                  </div>
                  <button className="btn btn-primary chat-small-btn" onClick={startNewChat}>New</button>
                </div>

                <div className="chat-session-list">
                  {isChatLoading && <div className="chat-side-note">Loading chats...</div>}
                  {!isChatLoading && chatSessions.length === 0 && <div className="chat-side-note">No chats yet.</div>}
                  {chatSessions.map((session) => (
                    <div key={session.id} className={`chat-session-item ${activeSessionId === session.id ? 'active' : ''}`}>
                      <button className="chat-session-btn" onClick={() => loadChatSession(session.id)}>
                        <span className="chat-session-title">{session.title}</span>
                        <span className="chat-session-preview">{session.last_preview || 'Open chat'}</span>
                      </button>
                      <button className="chat-session-delete" title="Delete chat" onClick={() => deleteChatSession(session.id)}>
                        {Icons.trash}
                      </button>
                    </div>
                  ))}
                </div>
              </aside>

              <section className="chat-main clean-chat-main">
                <div className="chat-main-header">
                  <div>
                    <div className="chat-main-title">Chat workspace</div>
                    <div className="chat-main-subtitle">
                      {geminiConfigured ? 'Grounded with recent analyses and video context from the backend.' : 'Gemini key is missing. Chat can open, but replies may fall back.'}
                    </div>
                  </div>
                  <div className="chat-header-actions">
                    <button className="btn" onClick={renameActiveChatSession} disabled={!activeSessionId}>Rename</button>
                    <button className="btn" onClick={downloadActiveChatSession} disabled={!activeSessionId}>Export Chat</button>
                    <button className="btn chat-close-btn" onClick={() => setShowChat(false)}>{Icons.close}</button>
                  </div>
                </div>

                <div className="chat-thread" ref={chatScrollRef}>
                  {chatMessages.length === 0 ? (
                    <div className="chat-empty">
                      <div className="chat-empty-card">
                        <div className="chat-empty-title">Start a new message</div>
                        <p>Ask a question, attach a video, compare it with recent analyses, or request a simpler explanation of the model output.</p>
                        <div className="action-chip-row chat-suggestions">
                          <button className="btn" onClick={() => seedChatPrompt('Why was the last clip uncertain?')}>Why uncertain?</button>
                          <button className="btn" onClick={() => seedChatPrompt('Summarize the recent analyses for me.')}>Summarize recent</button>
                          <button className="btn" onClick={() => seedChatPrompt('Compare this clip with the latest analysis once I attach it.')}>Compare clips</button>
                        </div>
                      </div>
                    </div>
                  ) : (
                    chatMessages.map((message) => (
                      <article key={message.id} className={`chat-bubble ${message.role}`}>
                        <div className="chat-bubble-role">{message.role === 'assistant' ? 'NeuralVision' : 'You'}</div>
                        <div className="chat-bubble-content">{message.content}</div>
                        {message.attachment && (
                          <div className="chat-attachment-chip">
                            {Icons.video} {message.attachment.filename} ({message.attachment.size_mb} MB)
                          </div>
                        )}
                        {message.video_context && (
                          <div className="chat-video-context">
                            <span>Video result</span>
                            <strong>{formatLabel(message.video_context.top_prediction)}</strong>
                            <span>{Number(message.video_context.top_confidence || 0).toFixed(1)}%</span>
                          </div>
                        )}
                      </article>
                    ))
                  )}
                </div>

                <div className="chat-composer sticky-composer">
                  {chatError && <div className="chat-error">{chatError}</div>}
                  {chatUpload && (
                    <div className="chat-upload-chip">
                      {Icons.video} {chatUpload.name}
                      <button className="chat-chip-close" onClick={() => setChatUpload(null)}>x</button>
                    </div>
                  )}
                  <div className="chat-input-row">
                    <button className="chat-attach-btn" onClick={() => chatFileInputRef.current.click()} title="Attach video">{Icons.attach}</button>
                    <textarea
                      className="chat-input"
                      value={chatInput}
                      onChange={(event) => setChatInput(event.target.value)}
                      placeholder="Type a message..."
                      rows={1}
                      onKeyDown={(event) => {
                        if (event.key === 'Enter' && !event.shiftKey) {
                          event.preventDefault();
                          sendChatMessage();
                        }
                      }}
                    />
                    <button className="chat-send-btn" onClick={sendChatMessage} disabled={isSendingChat || (!chatInput.trim() && !chatUpload)}>
                      {isSendingChat ? '...' : Icons.send}
                    </button>
                    <input
                      ref={chatFileInputRef}
                      type="file"
                      accept="video/*"
                      style={{ display: 'none' }}
                      onChange={(event) => {
                        const file = event.target.files[0];
                        if (!file) return;
                        if (!file.type.startsWith('video/')) {
                          setChatError('Only video files can be attached.');
                          return;
                        }
                        setChatUpload(file);
                        setChatError('');
                        event.target.value = '';
                      }}
                    />
                  </div>
                </div>
              </section>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function formatLabel(label) {
  return String(label || '-').replace(/_/g, ' ');
}

function formatDevice(device) {
  if (device === 'cuda') return 'GPU (CUDA)';
  if (device === 'cpu') return 'CPU';
  if (device === 'none') return 'No accelerator';
  return device || '-';
}

function formatTimestamp(value) {
  if (!value) return 'Unknown time';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return 'Unknown time';
  return date.toLocaleString([], {
    month: 'short',
    day: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
  });
}

function getWebcamErrorMessage(error) {
  const name = error?.name || '';

  if (name === 'NotAllowedError' || name === 'PermissionDeniedError') {
    return 'Camera permission was denied. Allow camera access in the browser and try again.';
  }
  if (name === 'NotFoundError' || name === 'DevicesNotFoundError') {
    return 'No camera was found on this device.';
  }
  if (name === 'NotReadableError' || name === 'TrackStartError') {
    return 'The camera is busy or blocked by another app. Close other apps using it and try again.';
  }
  if (name === 'OverconstrainedError' || name === 'ConstraintNotSatisfiedError') {
    return 'The requested camera settings are not supported on this device.';
  }
  if (name === 'SecurityError') {
    return 'Camera access was blocked by the browser security context. Open the app on localhost or HTTPS.';
  }
  if (name === 'AbortError') {
    return 'Camera startup was interrupted. Try again in a moment.';
  }

  return 'Webcam access failed. Check browser permissions and confirm the app is opened on localhost or HTTPS.';
}

function captureWebcamClip(stream, durationMs = 3000) {
  return new Promise((resolve, reject) => {
    try {
      if (typeof MediaRecorder === 'undefined') {
        reject(new Error('This browser cannot record webcam clips. Upload a video instead.'));
        return;
      }
      const preferredType = 'video/webm;codecs=vp8';
      const mimeType = MediaRecorder.isTypeSupported(preferredType) ? preferredType : 'video/webm';
      const recorder = new MediaRecorder(stream, { mimeType });
      const chunks = [];

      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) chunks.push(event.data);
      };
      recorder.onstop = () => {
        const blob = new Blob(chunks, { type: 'video/webm' });
        resolve(new File([blob], 'webcam_clip.webm', { type: 'video/webm' }));
      };
      recorder.onerror = () => reject(new Error('Recording failed.'));
      recorder.start();
      setTimeout(() => recorder.stop(), durationMs);
    } catch (error) {
      reject(error);
    }
  });
}

export default App;
