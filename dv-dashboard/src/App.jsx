import React, { useState, useMemo, memo } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar } from 'recharts';
import { AlertTriangle, TrendingUp, Shield, Clock, HelpCircle, Phone, Mail, Info } from 'lucide-react';

// Header Component with Logo
const AppHeader = ({ onHome, showBack = false }) => {
  return (
    <header className="bg-white border-b border-gray-200 sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-6 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            {showBack && (
              <button
                onClick={onHome}
                className="text-gray-600 hover:text-gray-800 transition-colors"
              >
                ← Back
              </button>
            )}
            <div className="flex items-center gap-3">
              <img 
                src="/sahaya-logo.png" 
                alt="SAHAYA Logo" 
                className="w-12 h-12"
                onError={(e) => {
                  e.target.style.display = 'none';
                }}
              />
              <div>
                <h1 className="text-2xl font-light text-gray-800">SAHAYA</h1>
                <p className="text-xs text-gray-500">Support • Protection • Aid</p>
              </div>
            </div>
          </div>
          <div className="flex items-center gap-2 text-sm text-gray-600">
            <Phone size={16} />
            <span className="hidden sm:inline">24/7 Helpline: 181</span>
          </div>
        </div>
      </div>
    </header>
  );
};

// Home Page Component
const HomePage = ({ onNavigate }) => {
  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-blue-50">
      <AppHeader onHome={() => {}} showBack={false} />
      
      <div className="flex items-center justify-center px-4 py-16">
        <div className="text-center max-w-3xl">
          {/* Logo */}
          <div className="mb-8 flex justify-center">
                  <img
          src="/sahaya-logo.png"
          alt="SAHAYA Logo"
          className="w-32 h-32"
          style={{ borderRadius: '50%' }}
          onError={(e) => {
            e.target.style.display = 'none';
          }}
        />
          </div>
          
          <h1 className="text-6xl font-light text-gray-800 mb-4">SAHAYA</h1>
          <p className="text-xl text-gray-600 mb-3">Support • Protection • Aid</p>
          <p className="text-base text-gray-500 mb-2">
            Domestic Violence Detection & Risk Assessment System
          </p>
          <p className="text-sm text-gray-400 mb-12 max-w-xl mx-auto">
            AI-powered early warning system to identify and assess domestic violence risks, 
            helping counselors provide timely intervention and support.
          </p>
          
          {/* Action Buttons */}
          <div className="flex flex-col sm:flex-row gap-4 justify-center mb-12">
            <button
              onClick={() => onNavigate('incremental')}
              className="px-8 py-4 bg-blue-600 text-white rounded-full text-lg font-medium hover:bg-blue-700 transition-all shadow-md hover:shadow-lg"
            >
              Start New Case Analysis
            </button>
            <button
              onClick={() => onNavigate('text')}
              className="px-8 py-4 bg-gray-600 text-white rounded-full text-lg font-medium hover:bg-gray-700 transition-all shadow-md hover:shadow-lg"
            >
              Batch Analysis
            </button>
            <button
              onClick={() => onNavigate('audio')}
              className="px-8 py-4 bg-purple-600 text-white rounded-full text-lg font-medium hover:bg-purple-700 transition-all shadow-md hover:shadow-lg"
            >
              Analyze Audio
            </button>
          </div>

          {/* Information Cards */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-16">
            <div className="bg-white rounded-lg p-6 shadow-sm border border-gray-100">
              <Shield className="text-blue-600 mb-3 mx-auto" size={32} />
              <h3 className="font-semibold text-gray-800 mb-2">Risk Assessment</h3>
              <p className="text-sm text-gray-600">
                AI-powered analysis identifies risk levels and escalation patterns to help prioritize cases.
              </p>
            </div>
            <div className="bg-white rounded-lg p-6 shadow-sm border border-gray-100">
              <TrendingUp className="text-orange-600 mb-3 mx-auto" size={32} />
              <h3 className="font-semibold text-gray-800 mb-2">Early Detection</h3>
              <p className="text-sm text-gray-600">
                Continuous monitoring detects warning signs before situations escalate to critical levels.
              </p>
            </div>
            <div className="bg-white rounded-lg p-6 shadow-sm border border-gray-100">
              <Clock className="text-green-600 mb-3 mx-auto" size={32} />
              <h3 className="font-semibold text-gray-800 mb-2">Timely Intervention</h3>
              <p className="text-sm text-gray-600">
                Real-time analysis enables counselors to provide immediate support when it's needed most.
              </p>
            </div>
          </div>

          {/* Help Section */}
          <div className="mt-12 bg-blue-50 rounded-lg p-6 border border-blue-100">
            <div className="flex items-start gap-3">
              <Info className="text-blue-600 flex-shrink-0 mt-1" size={20} />
              <div className="text-left">
                <h3 className="font-semibold text-gray-800 mb-2">How It Works</h3>
                <ul className="text-sm text-gray-600 space-y-1">
                  <li>• Add conversation messages or upload audio recordings</li>
                  <li>• AI analyzes patterns and identifies abuse types</li>
                  <li>• System calculates risk levels and recommends actions</li>
                  <li>• Counselors receive alerts for high-risk cases</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

// Incremental Text Analysis Page - allows adding messages one by one
const IncrementalTextAnalysis = ({ onBack, onResult }) => {
  const [caseId, setCaseId] = useState('');
  const [currentMessage, setCurrentMessage] = useState('');
  const [messageHistory, setMessageHistory] = useState([]);
  const [currentAnalysis, setCurrentAnalysis] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const analyzeMessages = async (messagesToAnalyze) => {
    setError('');
    setLoading(true);

    try {
      const response = await fetch('http://127.0.0.1:8000/analyze', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          case_id: caseId || 'CASE_' + Date.now(),
          messages: messagesToAnalyze
        }),
      });

      if (!response.ok) {
        throw new Error('Analysis failed. Please check your input.');
      }

      const data = await response.json();
      setCurrentAnalysis(data);
      return data;
    } catch (err) {
      setError(err.message || 'Failed to analyze. Please try again.');
      throw err;
    } finally {
      setLoading(false);
    }
  };

  const handleAddMessage = async (e) => {
    e.preventDefault();
    
    if (!currentMessage.trim()) {
      setError('Please enter a message');
      return;
    }

    const newMessage = {
      timestamp: new Date().toISOString(),
      text: currentMessage.trim()
    };

    const updatedHistory = [...messageHistory, newMessage];
    setMessageHistory(updatedHistory);
    setCurrentMessage('');

    // Automatically analyze after adding message
    await analyzeMessages(updatedHistory);
  };

  const handleViewDashboard = () => {
    if (currentAnalysis) {
      onResult(currentAnalysis);
    }
  };

  const handleReset = () => {
    setMessageHistory([]);
    setCurrentAnalysis(null);
    setCurrentMessage('');
    setError('');
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <AppHeader onHome={onBack} showBack={true} />
      <div className="max-w-6xl mx-auto px-4 py-8">

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left Column - Message Input */}
          <div className="lg:col-span-1">
            <div className="bg-white rounded-lg p-6 shadow-sm sticky top-20 border border-gray-100">
              <div className="flex items-center gap-2 mb-4">
                <Shield className="text-blue-600" size={24} />
                <h2 className="text-2xl font-light text-gray-800">New Case</h2>
              </div>
              
              {/* Help Tooltip */}
              <div className="mb-4 bg-blue-50 border border-blue-100 rounded-lg p-3">
                <div className="flex items-start gap-2">
                  <HelpCircle className="text-blue-600 flex-shrink-0 mt-0.5" size={16} />
                  <p className="text-xs text-gray-700">
                    Add messages one at a time. The system will analyze after each message to track risk progression.
                  </p>
                </div>
              </div>
              
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Case ID (optional)
                </label>
                <input
                  type="text"
                  value={caseId}
                  onChange={(e) => setCaseId(e.target.value)}
                  className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                  placeholder="Enter case ID"
                  disabled={messageHistory.length > 0}
                />
              </div>

              <form onSubmit={handleAddMessage} className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    New Message
                  </label>
                  <textarea
                    value={currentMessage}
                    onChange={(e) => setCurrentMessage(e.target.value)}
                    rows={4}
                    className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                    placeholder="Enter the conversation message..."
                    required
                  />
                </div>

                {error && (
                  <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg text-sm">
                    {error}
                  </div>
                )}

                <button
                  type="submit"
                  disabled={loading}
                  className="w-full px-6 py-3 bg-blue-600 text-white rounded-lg text-lg font-medium hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                >
                  {loading ? 'Analyzing...' : 'Add & Analyze'}
                </button>
              </form>

              {messageHistory.length > 0 && (
                <div className="mt-6 space-y-2">
                  <button
                    onClick={handleViewDashboard}
                    className="w-full px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors"
                  >
                    View Full Dashboard
                  </button>
                  <button
                    onClick={handleReset}
                    className="w-full px-4 py-2 bg-gray-200 text-gray-700 rounded-lg hover:bg-gray-300 transition-colors"
                  >
                    Reset Case
                  </button>
                </div>
              )}

              <div className="mt-6 pt-6 border-t border-gray-200">
                <p className="text-sm text-gray-600">
                  Messages added: <span className="font-semibold">{messageHistory.length}</span>
                </p>
              </div>
            </div>
          </div>

          {/* Right Column - Current Analysis */}
          <div className="lg:col-span-2">
            {currentAnalysis ? (
              <QuickAnalysisView analysis={currentAnalysis} />
            ) : (
              <div className="bg-white rounded-lg p-12 shadow-sm text-center">
                <Shield size={48} className="mx-auto mb-4 text-gray-400" />
                <p className="text-gray-600">Add your first message to begin analysis</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

// Memoized Chart Components to prevent re-renders
const TimelineChart = memo(({ timeline }) => {
  const data = useMemo(() => 
    timeline.map((msg, idx) => ({
      day: `Msg ${idx + 1}`,
      severity: msg.severity,
    })), [timeline]
  );

  return (
    <ResponsiveContainer width="100%" height={250}>
      <LineChart data={data}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="day" />
        <YAxis domain={[0, 5]} />
        <Tooltip />
        <Line 
          type="monotone" 
          dataKey="severity" 
          stroke="#ef4444" 
          strokeWidth={3}
          dot={{ fill: '#ef4444', r: 5 }}
        />
      </LineChart>
    </ResponsiveContainer>
  );
});

TimelineChart.displayName = 'TimelineChart';

const AbuseBreakdown = memo(({ classification }) => {
  const data = useMemo(() => 
    Object.entries(classification).map(([key, value]) => ({
      type: key.replace('_', ' '),
      probability: Math.round(value * 100)
    })), [classification]
  );

  return (
    <ResponsiveContainer width="100%" height={200}>
      <BarChart data={data} layout="vertical">
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis type="number" domain={[0, 100]} />
        <YAxis dataKey="type" type="category" width={100} />
        <Tooltip />
        <Bar dataKey="probability" fill="#3b82f6" />
      </BarChart>
    </ResponsiveContainer>
  );
});

AbuseBreakdown.displayName = 'AbuseBreakdown';

// Quick Analysis View - shows current state after each message
const QuickAnalysisView = memo(({ analysis }) => {
  const getRiskColor = (level) => {
    if (level === 'HIGH') return 'text-red-600 bg-red-50';
    if (level === 'MEDIUM') return 'text-yellow-600 bg-yellow-50';
    return 'text-green-600 bg-green-50';
  };

  // Memoize the latest classification to prevent unnecessary recalculations
  const latestClassification = useMemo(() => 
    analysis.timeline[analysis.timeline.length - 1]?.classification,
    [analysis.timeline]
  );

  return (
    <div className="space-y-6">
      {/* Risk Alert */}
      {analysis.flag_for_review && (
        <div className="bg-red-50 border-l-4 border-red-500 p-4 rounded-r-lg">
          <div className="flex items-start gap-3">
            <AlertTriangle className="text-red-500 flex-shrink-0" size={24} />
          <div>
              <p className="font-bold text-red-800 mb-1">⚠️ FLAGGED FOR REVIEW</p>
              <p className="text-sm text-red-700">{analysis.recommended_action}</p>
            </div>
          </div>
        </div>
      )}

      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-white rounded-lg shadow-sm p-4">
          <div className="text-sm text-gray-600 mb-1">Risk Level</div>
          <div className={`text-2xl font-bold ${getRiskColor(analysis.risk_level).split(' ')[0]}`}>
            {analysis.risk_level}
          </div>
        </div>
        <div className="bg-white rounded-lg shadow-sm p-4">
          <div className="text-sm text-gray-600 mb-1">Severity</div>
          <div className="text-2xl font-bold text-gray-800">
            {analysis.severity_latest.toFixed(1)}/5
          </div>
        </div>
        <div className="bg-white rounded-lg shadow-sm p-4">
          <div className="text-sm text-gray-600 mb-1">Escalation Risk</div>
          <div className="text-2xl font-bold text-gray-800">
            {(analysis.escalation_probability * 100).toFixed(0)}%
          </div>
        </div>
        <div className="bg-white rounded-lg shadow-sm p-4">
          <div className="text-sm text-gray-600 mb-1">Messages</div>
          <div className="text-2xl font-bold text-gray-800">
            {analysis.timeline.length}
          </div>
        </div>
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-white rounded-lg shadow-sm p-6">
          <h3 className="text-lg font-semibold text-gray-800 mb-4">Severity Trend</h3>
          <TimelineChart timeline={analysis.timeline} />
        </div>
        <div className="bg-white rounded-lg shadow-sm p-6">
          <h3 className="text-lg font-semibold text-gray-800 mb-4">Latest Message Analysis</h3>
          {latestClassification && <AbuseBreakdown classification={latestClassification} />}
        </div>
      </div>

      {/* Recommended Action */}
      <div className={`bg-white rounded-lg shadow-sm p-6 border-l-4 ${
        analysis.risk_level === 'HIGH' ? 'border-red-500' :
        analysis.risk_level === 'MEDIUM' ? 'border-yellow-500' :
        'border-green-500'
      }`}>
        <h3 className="text-lg font-semibold text-gray-800 mb-3">📋 Recommended Action</h3>
        <p className={`font-medium ${
          analysis.risk_level === 'HIGH' ? 'text-red-700' :
          analysis.risk_level === 'MEDIUM' ? 'text-yellow-700' :
          'text-green-700'
        }`}>
          {analysis.recommended_action}
        </p>
      </div>

      {/* Message Timeline */}
      <div className="bg-white rounded-lg shadow-sm p-6">
        <h3 className="text-lg font-semibold text-gray-800 mb-4">Message Timeline</h3>
        <div className="space-y-3">
          {analysis.timeline.map((item, index) => (
            <div 
              key={index} 
              className={`border-l-4 pl-4 py-3 ${
                item.severity > 3.5 ? 'border-red-500 bg-red-50' : 
                item.severity > 2 ? 'border-yellow-500 bg-yellow-50' : 
                'border-blue-500 bg-blue-50'
              }`}
            >
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2">
                  <Clock size={14} className="text-gray-500" />
                  <span className="text-xs text-gray-600">
                    {new Date(item.timestamp).toLocaleString()}
                  </span>
                </div>
                <span className={`px-2 py-1 rounded text-xs font-semibold ${
                  item.severity > 3.5 ? 'bg-red-500 text-white' : 
                  item.severity > 2 ? 'bg-yellow-500 text-white' : 
                  'bg-blue-500 text-white'
                }`}>
                  Severity: {item.severity.toFixed(1)}
                </span>
              </div>
              <p className="text-gray-800">{item.text}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
});

QuickAnalysisView.displayName = 'QuickAnalysisView';

// Full Dashboard View Component
const DashboardView = ({ result, onBack, onAddMore }) => {
  const getRiskColor = (level) => {
    if (level === 'HIGH') return 'bg-red-500';
    if (level === 'MEDIUM') return 'bg-yellow-500';
    return 'bg-green-500';
  };

  const getAbuseTypeColor = (type) => {
    const colors = {
      control: 'bg-blue-400',
      verbal: 'bg-yellow-400',
      threat: 'bg-orange-400',
      physical: 'bg-red-500',
      severe_physical: 'bg-red-800'
    };
    return colors[type] || 'bg-gray-400';
  };

  const TimelineChart = ({ timeline }) => {
    const data = timeline.map((msg, idx) => ({
      day: `Msg ${idx + 1}`,
      severity: msg.severity,
    }));

    return (
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="day" />
          <YAxis domain={[0, 5]} label={{ value: 'Severity', angle: -90, position: 'insideLeft' }} />
          <Tooltip />
          <Legend />
          <Line 
            type="monotone" 
            dataKey="severity" 
            stroke="#ef4444" 
            strokeWidth={3}
            dot={{ fill: '#ef4444', r: 6 }}
          />
        </LineChart>
      </ResponsiveContainer>
    );
  };

  const AbuseBreakdown = ({ classification }) => {
    const data = Object.entries(classification).map(([key, value]) => ({
      type: key.replace('_', ' '),
      probability: Math.round(value * 100)
    }));

    return (
      <ResponsiveContainer width="100%" height={250}>
        <BarChart data={data} layout="vertical">
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis type="number" domain={[0, 100]} />
          <YAxis dataKey="type" type="category" width={100} />
          <Tooltip />
          <Bar dataKey="probability" fill="#3b82f6" />
        </BarChart>
      </ResponsiveContainer>
    );
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <AppHeader onHome={onBack} showBack={true} />
      
      {/* Dashboard Header Bar */}
      <div className="bg-gradient-to-r from-blue-600 to-purple-600 text-white">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold">Case Analysis Dashboard</h1>
              <p className="text-blue-100 text-sm">Risk Assessment & Intervention Recommendations</p>
            </div>
            <button
              onClick={onAddMore}
              className="bg-white/20 hover:bg-white/30 rounded-lg px-4 py-2 transition-colors text-sm font-medium"
            >
              + Add More Messages
            </button>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-6 py-6">
        {/* Risk Alert Banner */}
        {result.flag_for_review && (
          <div className="bg-red-50 border-l-4 border-red-500 p-6 mb-6 rounded-r-lg shadow-md">
            <div className="flex items-start gap-4">
              <AlertTriangle className="text-red-500 flex-shrink-0" size={32} />
              <div className="flex-1">
                <h2 className="text-xl font-bold text-red-800 mb-2">⚠️ FLAGGED FOR IMMEDIATE REVIEW</h2>
                <p className="text-red-700 mb-3">{result.recommended_action}</p>
              </div>
              </div>
            </div>
          )}

        {/* Key Metrics */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-6">
          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center justify-between mb-2">
              <span className="text-gray-600 text-sm">Risk Level</span>
              <TrendingUp className="text-red-500" size={20} />
            </div>
            <p className={`text-3xl font-bold ${result.risk_level === 'HIGH' ? 'text-red-600' : result.risk_level === 'MEDIUM' ? 'text-yellow-600' : 'text-green-600'}`}>
              {result.risk_level}
            </p>
            <div className="mt-2">
              <div className={`h-2 rounded-full ${getRiskColor(result.risk_level)}`} 
                   style={{ width: `${Math.min((result.risk_score / 3) * 100, 100)}%` }} />
            </div>
          </div>

          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center justify-between mb-2">
              <span className="text-gray-600 text-sm">Escalation Risk</span>
              <AlertTriangle className="text-orange-500" size={20} />
            </div>
            <p className="text-3xl font-bold text-gray-800">
              {(result.escalation_probability * 100).toFixed(0)}%
            </p>
            <p className="text-sm text-gray-600 mt-2">
              {result.escalation_probability > 0.6 ? 'High probability' : result.escalation_probability > 0.3 ? 'Moderate probability' : 'Low probability'}
            </p>
            </div>

          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center justify-between mb-2">
              <span className="text-gray-600 text-sm">Current Severity</span>
              <Shield className="text-red-500" size={20} />
            </div>
            <p className="text-3xl font-bold text-gray-800">
              {result.severity_latest.toFixed(1)}/5
            </p>
            <p className="text-sm text-gray-600 mt-2">Latest message severity</p>
          </div>

          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center justify-between mb-2">
              <span className="text-gray-600 text-sm">Timeline</span>
              <Clock className="text-blue-500" size={20} />
            </div>
            <p className="text-3xl font-bold text-gray-800">
              {result.timeline.length}
            </p>
            <p className="text-sm text-gray-600 mt-2">Messages analyzed</p>
          </div>
        </div>

        {/* Main Content Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Timeline Section */}
          <div className="lg:col-span-2 space-y-6">
            {/* Severity Trend */}
            <div className="bg-white rounded-lg shadow p-6">
              <h3 className="text-lg font-bold text-gray-800 mb-4">Abuse Severity Trend</h3>
              <TimelineChart timeline={result.timeline} />
            </div>

            {/* Message Timeline */}
            <div className="bg-white rounded-lg shadow p-6">
              <h3 className="text-lg font-bold text-gray-800 mb-4">Message Timeline</h3>
              <div className="space-y-4">
                {result.timeline.map((msg, idx) => (
                  <div 
                    key={idx} 
                    className={`border-l-4 pl-4 py-3 ${
                      msg.severity > 3.5 ? 'border-red-500 bg-red-50' : 
                      msg.severity > 2 ? 'border-yellow-500 bg-yellow-50' : 
                      'border-blue-500 bg-blue-50'
                    }`}
                  >
                    <div className="flex items-start justify-between mb-2">
                      <div className="flex items-center gap-2">
                        <Clock size={14} className="text-gray-500" />
                        <span className="text-sm text-gray-600">
                          {new Date(msg.timestamp).toLocaleString()}
                        </span>
                      </div>
                      <span className={`px-3 py-1 rounded-full text-xs font-semibold ${
                        msg.severity > 3.5 ? 'bg-red-500 text-white' : 
                        msg.severity > 2 ? 'bg-yellow-500 text-white' : 
                        'bg-blue-500 text-white'
                      }`}>
                        Severity: {msg.severity.toFixed(1)}
                      </span>
                    </div>
                    <p className="text-gray-800 font-medium">{msg.text}</p>
                    
                    {/* Abuse type indicators */}
                    <div className="flex gap-2 mt-3 flex-wrap">
                      {Object.entries(msg.classification).map(([type, prob]) => 
                        prob > 0.3 && (
                          <span key={type} 
                                className={`px-2 py-1 rounded text-xs font-semibold text-white ${getAbuseTypeColor(type)}`}>
                            {type.replace('_', ' ')}: {(prob * 100).toFixed(0)}%
                          </span>
                        )
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Analysis Panel */}
          <div className="space-y-6">
            {/* Latest Classification */}
            <div className="bg-white rounded-lg shadow p-6">
              <h3 className="text-lg font-bold text-gray-800 mb-4">Latest Message Analysis</h3>
              <AbuseBreakdown classification={result.timeline[result.timeline.length - 1].classification} />
            </div>

            {/* Recommended Actions */}
            <div className="bg-white rounded-lg shadow p-6">
              <h3 className="text-lg font-bold text-gray-800 mb-3">📋 Recommended Action</h3>
              <div className="space-y-3">
                <div className={`p-4 rounded-lg ${
                  result.risk_level === 'HIGH' ? 'bg-red-50 border border-red-200' :
                  result.risk_level === 'MEDIUM' ? 'bg-yellow-50 border border-yellow-200' :
                  'bg-green-50 border border-green-200'
                }`}>
                  <p className={`font-semibold ${
                    result.risk_level === 'HIGH' ? 'text-red-800' :
                    result.risk_level === 'MEDIUM' ? 'text-yellow-800' :
                    'text-green-800'
                  }`}>
                    {result.recommended_action}
                  </p>
                </div>
                <div className="space-y-2">
                  {result.risk_level === 'HIGH' && (
                    <>
                      <button className="w-full bg-red-600 text-white py-2 rounded-lg hover:bg-red-700 transition-colors text-sm font-semibold">
                        Immediate Safety Planning
                      </button>
                      <button className="w-full bg-orange-600 text-white py-2 rounded-lg hover:bg-orange-700 transition-colors text-sm">
                        Emergency Contact
                      </button>
                    </>
                  )}
                  {result.risk_level === 'MEDIUM' && (
                    <>
                      <button className="w-full bg-yellow-600 text-white py-2 rounded-lg hover:bg-yellow-700 transition-colors text-sm font-semibold">
                        Counselor Outreach
                      </button>
                      <button className="w-full bg-blue-600 text-white py-2 rounded-lg hover:bg-blue-700 transition-colors text-sm">
                        Schedule Follow-up
                      </button>
                    </>
                  )}
                  {result.risk_level === 'LOW' && (
                    <>
                      <button className="w-full bg-green-600 text-white py-2 rounded-lg hover:bg-green-700 transition-colors text-sm font-semibold">
                        Continue Monitoring
                      </button>
                      <button className="w-full bg-blue-600 text-white py-2 rounded-lg hover:bg-blue-700 transition-colors text-sm">
                        Regular Check-in
                      </button>
                    </>
                  )}
                  <button className="w-full bg-gray-600 text-white py-2 rounded-lg hover:bg-gray-700 transition-colors text-sm">
                    View Resources
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

// Analyze Text Page Component (Batch mode - original)
const AnalyzeTextPage = ({ onBack, onResult }) => {
  const [caseId, setCaseId] = useState('');
  const [messages, setMessages] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    try {
      let parsedMessages;
      
      try {
        parsedMessages = JSON.parse(messages);
        if (!Array.isArray(parsedMessages)) {
          throw new Error('JSON must be an array');
        }
      } catch {
        const lines = messages.trim().split('\n').filter(line => line.trim());
        parsedMessages = lines.map(line => {
          const parts = line.split('|');
          if (parts.length >= 2) {
            return {
              timestamp: parts[0].trim(),
              text: parts.slice(1).join('|').trim()
            };
          } else {
            return {
              timestamp: new Date().toISOString(),
              text: line.trim()
            };
          }
        });
      }

      if (parsedMessages.length === 0) {
        throw new Error('Please provide at least one message');
      }

      const response = await fetch('http://127.0.0.1:8000/analyze', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          case_id: caseId || 'CASE_' + Date.now(),
          messages: parsedMessages
        }),
      });

      if (!response.ok) {
        throw new Error('Analysis failed. Please check your input.');
      }

      const data = await response.json();
      onResult(data);
    } catch (err) {
      setError(err.message || 'Failed to analyze. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <AppHeader onHome={onBack} showBack={true} />
      <div className="flex items-center justify-center px-4 py-12">
        <div className="w-full max-w-2xl">
          <div className="bg-white rounded-lg p-8 shadow-sm border border-gray-100">
            <div className="flex items-center gap-3 mb-6">
              <Shield className="text-blue-600" size={32} />
              <h2 className="text-3xl font-light text-gray-800">Batch Text Analysis</h2>
            </div>
            
            {/* Help Section */}
            <div className="mb-6 bg-blue-50 border border-blue-100 rounded-lg p-4">
              <div className="flex items-start gap-2">
                <Info className="text-blue-600 flex-shrink-0 mt-0.5" size={18} />
                <div className="text-sm text-gray-700">
                  <p className="font-semibold mb-1">Upload multiple messages at once:</p>
                  <ul className="list-disc list-inside space-y-1 text-xs">
                    <li>JSON array format (recommended)</li>
                    <li>Pipe-separated: timestamp|message</li>
                    <li>Simple lines (timestamps auto-generated)</li>
                  </ul>
                </div>
              </div>
            </div>
          
          <form onSubmit={handleSubmit} className="space-y-6">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Case ID (optional)
              </label>
              <input
                type="text"
                value={caseId}
                onChange={(e) => setCaseId(e.target.value)}
                className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                placeholder="Enter case ID or leave blank"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Conversation Messages
              </label>
              <textarea
                value={messages}
                onChange={(e) => setMessages(e.target.value)}
                rows={12}
                className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 font-mono text-sm"
                placeholder={`Format 1 (JSON array):
[
  {"timestamp": "2024-01-01T10:00:00Z", "text": "First message"},
  {"timestamp": "2024-01-01T10:05:00Z", "text": "Second message"}
]

Format 2 (line-by-line with pipe):
2024-01-01T10:00:00Z|First message
2024-01-01T10:05:00Z|Second message

Format 3 (simple lines, timestamps auto-generated):
First message
Second message`}
                required
              />
            </div>

            {error && (
              <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg">
                {error}
              </div>
            )}

            <button
              type="submit"
              disabled={loading}
              className="w-full px-6 py-3 bg-blue-600 text-white rounded-full text-lg font-medium hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              {loading ? 'Analyzing...' : 'Submit for Analysis'}
            </button>
          </form>
          </div>
        </div>
      </div>
    </div>
  );
};

// Analyze Audio Page Component
const AnalyzeAudioPage = ({ onBack, onResult }) => {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
    setError('');
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!file) {
      setError('Please select an audio file');
      return;
    }

    setError('');
    setLoading(true);

    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch('http://127.0.0.1:8000/speech-analyze', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Analysis failed. Please check your audio file.');
      }

      const data = await response.json();
      onResult(data.analysis);
    } catch (err) {
      setError(err.message || 'Failed to analyze audio. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <AppHeader onHome={onBack} showBack={true} />
      <div className="flex items-center justify-center px-4 py-12">
        <div className="w-full max-w-2xl">
          <div className="bg-white rounded-lg p-8 shadow-sm border border-gray-100">
            <div className="flex items-center gap-3 mb-6">
              <Shield className="text-purple-600" size={32} />
              <h2 className="text-3xl font-light text-gray-800">Audio Analysis</h2>
            </div>
            
            {/* Help Section */}
            <div className="mb-6 bg-purple-50 border border-purple-100 rounded-lg p-4">
              <div className="flex items-start gap-2">
                <Info className="text-purple-600 flex-shrink-0 mt-0.5" size={18} />
                <div className="text-sm text-gray-700">
                  <p className="font-semibold mb-1">Upload audio recordings:</p>
                  <p className="text-xs">The system will transcribe the audio and analyze it for domestic violence indicators. Supports common audio formats (MP3, WAV, OGG, etc.)</p>
                </div>
              </div>
            </div>
          
          <form onSubmit={handleSubmit} className="space-y-6">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Audio File
              </label>
              <input
                type="file"
                accept="audio/*"
                onChange={handleFileChange}
                className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
              {file && (
                <p className="mt-2 text-sm text-gray-600">
                  Selected: {file.name}
                </p>
              )}
            </div>

            {error && (
              <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg">
                {error}
              </div>
            )}

            <button
              type="submit"
              disabled={loading || !file}
              className="w-full px-6 py-3 bg-purple-600 text-white rounded-full text-lg font-medium hover:bg-purple-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              {loading ? 'Analyzing...' : 'Submit for Analysis'}
            </button>
          </form>
          </div>
        </div>
      </div>
    </div>
  );
};

// Main App Component
const App = () => {
  const [currentPage, setCurrentPage] = useState('home');
  const [result, setResult] = useState(null);

  const handleNavigate = (page) => {
    setCurrentPage(page);
    setResult(null);
  };

  const handleResult = (data) => {
    setResult(data);
    setCurrentPage('dashboard');
  };

  const handleBack = () => {
    setCurrentPage('home');
    setResult(null);
  };

  const handleAddMore = () => {
    setCurrentPage('incremental');
    // Keep the result data for context if needed
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {currentPage === 'home' && <HomePage onNavigate={handleNavigate} />}
      {currentPage === 'text' && (
        <AnalyzeTextPage onBack={handleBack} onResult={handleResult} />
      )}
      {currentPage === 'incremental' && (
        <IncrementalTextAnalysis onBack={handleBack} onResult={handleResult} />
      )}
      {currentPage === 'audio' && (
        <AnalyzeAudioPage onBack={handleBack} onResult={handleResult} />
      )}
      {currentPage === 'dashboard' && result && (
        <DashboardView result={result} onBack={handleBack} onAddMore={handleAddMore} />
      )}
    </div>
  );
};

export default App;
