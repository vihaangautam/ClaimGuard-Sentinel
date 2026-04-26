import React from 'react';
import { 
  ShieldCheck, 
  BrainCircuit, 
  Map as MapIcon, 
  BellRing, 
  PieChart, 
  Satellite, 
  CloudOff, 
  Zap,
  ArrowRight,
  BarChart3,
  Globe2,
  CheckCircle2,
  Layers,
  Activity,
  FileText,
  Users,
  AlertTriangle
} from 'lucide-react';

const Navbar = ({ onExplore }) => (
  <nav className="sticky top-0 z-50 w-full backdrop-blur-lg bg-white/80 border-b border-slate-200">
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
      <div className="flex justify-between items-center h-20">
        <div className="flex items-center gap-2">
          <div className="bg-emerald-600 p-2 rounded-lg">
            <Satellite className="w-6 h-6 text-white" />
          </div>
          <span className="font-bold text-xl text-slate-900 tracking-tight">ClaimGuard<span className="text-emerald-600">Sentinel</span></span>
        </div>
        <div className="hidden md:flex space-x-8">
          <a href="#problem" className="text-slate-600 hover:text-emerald-600 font-medium transition-colors">The Challenge</a>
          <a href="#features" className="text-slate-600 hover:text-emerald-600 font-medium transition-colors">Platform</a>
          <a href="#technology" className="text-slate-600 hover:text-emerald-600 font-medium transition-colors">AI & Tech</a>
        </div>
        <div className="flex items-center space-x-4">
          <button onClick={onExplore} className="bg-slate-900 hover:bg-slate-800 text-white px-5 py-2.5 rounded-lg font-medium transition-all shadow-sm hover:shadow-md flex items-center gap-2">
            Launch Command Center
            <ArrowRight className="w-4 h-4" />
          </button>
        </div>
      </div>
    </div>
  </nav>
);

const Hero = ({ onExplore }) => (
  <section className="relative pt-20 pb-32 overflow-hidden bg-slate-50">
    <div className="absolute inset-0 bg-[url('https://www.transparenttextures.com/patterns/cubes.png')] opacity-5"></div>
    <div className="absolute top-0 right-0 -translate-y-12 translate-x-1/3 w-[800px] h-[800px] bg-emerald-100 rounded-full blur-3xl opacity-50 pointer-events-none"></div>
    
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative">
      <div className="grid lg:grid-cols-2 gap-12 lg:gap-8 items-center">
        
        {/* Left Content */}
        <div className="max-w-2xl">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-emerald-100 text-emerald-700 font-medium text-sm mb-6 border border-emerald-200">
            <Activity className="w-4 h-4" />
            <span>Built for Insurance Investigators & Risk Managers</span>
          </div>
          <h1 className="text-5xl lg:text-6xl font-extrabold text-slate-900 tracking-tight leading-[1.15] mb-6">
            Proactive Drought Risk & <span className="text-transparent bg-clip-text bg-gradient-to-r from-emerald-600 to-teal-500">Fraud Prevention</span>
          </h1>
          <p className="text-lg text-slate-600 mb-8 leading-relaxed">
            Eliminate subjective field surveys and 18-month settlement delays. 
            ClaimGuard Sentinel empowers agricultural insurers to verify claims in under a second and forecast drought conditions 3 months in advance using real-time satellite imagery.
          </p>
          <div className="flex flex-col sm:flex-row gap-4">
            <button onClick={onExplore} className="bg-emerald-600 hover:bg-emerald-700 text-white px-8 py-3.5 rounded-xl font-semibold text-lg transition-all shadow-lg shadow-emerald-600/20 flex items-center justify-center gap-2">
              Explore Dashboard
            </button>
            <a href="/Research_Paper.pdf" target="_blank" rel="noreferrer" className="bg-white border-2 border-slate-200 hover:border-slate-300 text-slate-700 px-8 py-3.5 rounded-xl font-semibold text-lg transition-all flex items-center justify-center gap-2 shadow-sm">
              <FileText className="w-5 h-5" />
              Read Research Paper
            </a>
          </div>
          
          <div className="mt-10 flex items-center gap-6 text-sm text-slate-500 font-medium flex-wrap">
            <div className="flex items-center gap-2"><CheckCircle2 className="w-5 h-5 text-emerald-500" /> &lt;1s Verification</div>
            <div className="flex items-center gap-2"><CheckCircle2 className="w-5 h-5 text-emerald-500" /> Objective Satellite Data</div>
            <div className="flex items-center gap-2"><CheckCircle2 className="w-5 h-5 text-emerald-500" /> Multi-Spectral Analytics</div>
          </div>
        </div>

        {/* Right Abstract Dashboard Mockup */}
        <div className="relative mx-auto w-full max-w-lg lg:max-w-none">
          <div className="relative rounded-2xl bg-slate-900 border border-slate-800 shadow-2xl shadow-slate-900/50 p-3 z-10 transform lg:rotate-[-2deg] transition-transform hover:rotate-0 duration-500">
            {/* Mockup Header */}
            <div className="flex items-center justify-between border-b border-slate-800 pb-3 px-3 mb-4">
              <div className="flex gap-2">
                <div className="w-3 h-3 rounded-full bg-slate-700 hover:bg-red-500 transition-colors"></div>
                <div className="w-3 h-3 rounded-full bg-slate-700 hover:bg-amber-500 transition-colors"></div>
                <div className="w-3 h-3 rounded-full bg-slate-700 hover:bg-emerald-500 transition-colors"></div>
              </div>
              <div className="text-xs font-semibold text-slate-400">South India • Real-time Monitoring</div>
            </div>
            
            {/* Mockup Map Area */}
            <div className="relative bg-[#0f172a] rounded-xl h-64 mb-4 border border-slate-800 overflow-hidden">
              <div className="absolute inset-0 opacity-20 pointer-events-none flex items-center justify-center">
                <Globe2 className="text-emerald-500 w-[120%] h-[120%] -right-10 -bottom-20 stroke-1" />
              </div>
              
              {/* Simulated Map Nodes */}
              {/* Anantapur - High Risk */}
              <div className="absolute top-[30%] left-[35%] group">
                <div className="absolute -inset-2 bg-red-500/20 rounded-full animate-ping"></div>
                <div className="relative w-4 h-4 bg-red-500 rounded-full border-[3px] border-[#0f172a] z-10 shadow-[0_0_10px_rgba(239,68,68,0.8)]"></div>
                <div className="absolute -top-6 left-1/2 -translate-x-1/2 bg-slate-800 text-[10px] text-white px-2 py-0.5 rounded border border-slate-700 whitespace-nowrap opacity-100">Anantapur</div>
              </div>

              {/* Kurnool - Warning */}
              <div className="absolute top-[45%] left-[55%] group">
                <div className="absolute -inset-1.5 bg-amber-500/20 rounded-full animate-ping delay-100 duration-1000"></div>
                <div className="relative w-3.5 h-3.5 bg-amber-500 rounded-full border-[3px] border-[#0f172a] z-10"></div>
              </div>

              {/* Dharmapuri - Safe */}
              <div className="absolute bottom-[35%] right-[30%] group">
                <div className="absolute -inset-1 bg-emerald-500/20 rounded-full animate-ping delay-300 duration-1000"></div>
                <div className="relative w-3 h-3 bg-emerald-500 rounded-full border-2 border-[#0f172a] z-10"></div>
              </div>

              {/* Koppal - Warning */}
              <div className="absolute top-[20%] right-[40%] group">
                <div className="absolute -inset-2 bg-amber-500/20 rounded-full animate-ping delay-200"></div>
                <div className="relative w-4 h-4 bg-amber-500 rounded-full border-[3px] border-[#0f172a] z-10 shadow-[0_0_10px_rgba(245,158,11,0.5)]"></div>
              </div>
              
              {/* Map Controls & Legends Overlay */}
              <div className="absolute top-4 right-4 bg-slate-900/80 backdrop-blur-md shadow-sm p-2.5 rounded-lg text-[10px] font-medium border border-slate-700/50 text-slate-300">
                <div className="flex items-center gap-2 mb-1.5"><div className="w-2 h-2 rounded-full bg-red-500 shadow-[0_0_5px_rgba(239,68,68,0.8)]"></div> High Risk</div>
                <div className="flex items-center gap-2 mb-1.5"><div className="w-2 h-2 rounded-full bg-amber-500"></div> Warning</div>
                <div className="flex items-center gap-2"><div className="w-2 h-2 rounded-full bg-emerald-500"></div> Safe</div>
              </div>
              
              <div className="absolute bottom-4 left-4 right-4 bg-slate-900/80 backdrop-blur-md shadow-sm p-3 rounded-lg border border-slate-700/50 flex justify-between items-center">
                 <div className="text-xs font-bold text-slate-200 flex items-center gap-2">
                   <Activity className="w-4 h-4 text-emerald-500" />
                   Timeline: Forecast Mode
                 </div>
                 <div className="flex gap-1">
                   <div className="w-8 h-1.5 rounded-full bg-emerald-500"></div>
                   <div className="w-8 h-1.5 rounded-full bg-emerald-400"></div>
                   <div className="w-8 h-1.5 rounded-full bg-amber-400"></div>
                   <div className="w-8 h-1.5 rounded-full bg-slate-700"></div>
                 </div>
              </div>
            </div>
            
            {/* Mockup Cards */}
            <div className="grid grid-cols-2 gap-3">
              <div className="p-4 rounded-xl border border-slate-700/50 bg-slate-800/50 hover:bg-slate-800 transition-colors">
                <div className="flex justify-between items-start mb-2">
                  <div className="text-xs font-bold text-slate-400">NDVI Forecast</div>
                  <BarChart3 className="w-4 h-4 text-emerald-400" />
                </div>
                <div className="text-2xl font-black text-white">0.42</div>
                <div className="text-[10px] font-semibold text-emerald-400 mt-1 bg-emerald-500/10 px-2 py-0.5 rounded inline-block">Stable Trend</div>
              </div>
              <div className="p-4 rounded-xl border border-red-900/30 bg-red-950/20 hover:bg-red-900/20 transition-colors">
                <div className="flex justify-between items-start mb-2">
                  <div className="text-xs font-bold text-red-400">Fraud Alerts</div>
                  <ShieldCheck className="w-4 h-4 text-red-400" />
                </div>
                <div className="text-2xl font-black text-white">12</div>
                <div className="text-[10px] font-semibold text-red-400 mt-1 bg-red-500/10 px-2 py-0.5 rounded inline-block">Requires Review</div>
              </div>
            </div>
          </div>
          
          {/* Decorative elements behind mockup */}
          <div className="absolute top-1/2 -right-8 -translate-y-1/2 w-24 h-24 bg-teal-100 rounded-full blur-2xl z-0"></div>
          <div className="absolute -bottom-6 -left-6 w-32 h-32 bg-emerald-100 rounded-full blur-2xl z-0"></div>
        </div>
      </div>
    </div>
  </section>
);

const ProblemContext = () => (
  <section id="problem" className="py-24 bg-white relative">
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
      <div className="text-center max-w-3xl mx-auto mb-16">
        <h2 className="text-emerald-600 font-bold tracking-wide uppercase text-sm mb-3">The Challenge</h2>
        <h3 className="text-3xl md:text-4xl font-extrabold text-slate-900 mb-6">Why Agricultural Insurance Needs an Upgrade</h3>
        <p className="text-lg text-slate-600">
          Traditional crop insurance relies on slow, manual, and subjective methods, leading to rampant fraud and massive delays for genuine farmers.
        </p>
      </div>

      <div className="grid md:grid-cols-3 gap-8">
        <div className="bg-slate-50 rounded-2xl p-8 border border-slate-200">
          <div className="w-12 h-12 bg-red-100 text-red-600 rounded-xl flex items-center justify-center mb-6">
            <AlertTriangle className="w-6 h-6" />
          </div>
          <h4 className="text-xl font-bold text-slate-900 mb-3">Rampant Claim Fraud</h4>
          <p className="text-slate-600 text-sm leading-relaxed">
            Intermediaries routinely file drought claims for unaffected areas. Without objective real-time data, insurance companies struggle to separate genuine distress from fraudulent submissions.
          </p>
        </div>
        
        <div className="bg-slate-50 rounded-2xl p-8 border border-slate-200">
          <div className="w-12 h-12 bg-amber-100 text-amber-600 rounded-xl flex items-center justify-center mb-6">
            <Users className="w-6 h-6" />
          </div>
          <h4 className="text-xl font-bold text-slate-900 mb-3">Subjective Field Reports</h4>
          <p className="text-slate-600 text-sm leading-relaxed">
            Risk assessment currently relies on manual Crop Cutting Experiments (CCEs). These are highly localized, extremely labor-intensive, and prone to human error and manipulation.
          </p>
        </div>

        <div className="bg-slate-50 rounded-2xl p-8 border border-slate-200">
          <div className="w-12 h-12 bg-blue-100 text-blue-600 rounded-xl flex items-center justify-center mb-6">
            <Zap className="w-6 h-6" />
          </div>
          <h4 className="text-xl font-bold text-slate-900 mb-3">Reactive Operations</h4>
          <p className="text-slate-600 text-sm leading-relaxed">
            Insurance officers only discover drought severity after claims pour in. The lack of proactive forecasting depletes financial reserves unexpectedly and delays payouts by 12–18 months.
          </p>
        </div>
      </div>
    </div>
  </section>
);

const FeatureCard = ({ icon: Icon, title, description, badge }) => (
  <div className="bg-white p-8 rounded-2xl border border-slate-100 shadow-sm hover:shadow-xl transition-all duration-300 relative overflow-hidden group">
    <div className="absolute top-0 right-0 p-6 opacity-5 group-hover:opacity-10 transition-opacity transform translate-x-4 -translate-y-4">
      <Icon className="w-32 h-32" />
    </div>
    <div className="w-14 h-14 bg-emerald-50 text-emerald-600 rounded-xl flex items-center justify-center mb-6 border border-emerald-100 group-hover:bg-emerald-600 group-hover:text-white transition-colors">
      <Icon className="w-7 h-7" />
    </div>
    {badge && <span className="absolute top-8 right-8 text-[10px] font-bold uppercase tracking-wider bg-slate-100 text-slate-600 py-1 px-2 rounded">{badge}</span>}
    <h3 className="text-xl font-bold text-slate-900 mb-3">{title}</h3>
    <p className="text-slate-600 leading-relaxed text-sm">
      {description}
    </p>
  </div>
);

const Features = () => {
  const features = [
    {
      icon: ShieldCheck,
      title: "Automated Verification Engine",
      description: "Instantly cross-reference claim dates & locations against historical satellite data (NDVI, SMI, Rainfall). Categorizes claims as APPROVED, FLAGGED, or REJECTED with a confidence score.",
      badge: "< 1 Second"
    },
    {
      icon: BrainCircuit,
      title: "Deep Learning Forecasting",
      description: "Our fine-tuned CNN-LSTM hybrid model captures cross-feature correlations and temporal dependencies to predict vegetation health (NDVI) up to 3 months into the future.",
      badge: "CNN-LSTM"
    },
    {
      icon: MapIcon,
      title: "Real-Time Risk Dashboard",
      description: "Interactive mapping of 13 drought-prone districts. Use the Time-Travel Date Picker to scrub through historical data (Jan 2019+) or view futuristic AI predictions seamlessly.",
      badge: "Leaflet UI"
    },
    {
      icon: PieChart,
      title: "K-Means Cluster Analytics",
      description: "Optimize risk exposure for portfolio diversification. Groups districts into Severe, Moderate, and Resilient zones based on 6 key metrics, preventing over-concentration.",
      badge: "Risk Mgmt"
    }
  ];

  return (
    <section id="features" className="py-24 bg-slate-50 relative border-t border-slate-100">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center max-w-3xl mx-auto mb-16">
          <h2 className="text-emerald-600 font-bold tracking-wide uppercase text-sm mb-3">Core Capabilities</h2>
          <h3 className="text-3xl md:text-4xl font-extrabold text-slate-900 mb-6">Moving from Reactive to Proactive</h3>
          <p className="text-lg text-slate-600">
            Replace subjective field reports with objective, multi-spectral satellite data. ClaimGuard Sentinel modernizes every aspect of agricultural insurance operations.
          </p>
        </div>
        
        <div className="grid md:grid-cols-2 gap-8">
          {features.map((feat, idx) => (
            <FeatureCard key={idx} {...feat} />
          ))}
        </div>
      </div>
    </section>
  );
};

const Technology = () => (
  <section id="technology" className="py-24 bg-slate-900 text-white relative overflow-hidden">
    {/* Abstract Background */}
    <div className="absolute inset-0 opacity-10">
      <div className="absolute top-0 -left-1/4 w-1/2 h-full bg-emerald-500 blur-[150px] rounded-full mix-blend-screen"></div>
      <div className="absolute bottom-0 -right-1/4 w-1/2 h-full bg-blue-500 blur-[150px] rounded-full mix-blend-screen"></div>
    </div>

    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10">
      <div className="grid lg:grid-cols-2 gap-16 items-center">
        <div>
          <h2 className="text-emerald-400 font-bold tracking-wide uppercase text-sm mb-3">Technical Architecture</h2>
          <h3 className="text-3xl md:text-4xl font-extrabold mb-6">Built for Cloud-Resistant Reliability</h3>
          <p className="text-slate-300 text-lg mb-8 leading-relaxed">
            Standard optical satellites fail during monsoon seasons. ClaimGuard Sentinel bypasses complex cloud removal pipelines by utilizing microwave radar (SMAP) and monthly composites (MODIS) to ensure uninterrupted monitoring.
          </p>
          
          <div className="space-y-6">
            <div className="flex gap-4">
              <div className="flex-shrink-0 w-12 h-12 rounded-xl bg-slate-800 border border-slate-700 flex items-center justify-center">
                <DatabaseIcon className="w-6 h-6 text-emerald-400" />
              </div>
              <div>
                <h4 className="text-xl font-bold mb-1">Google Earth Engine Feeds</h4>
                <p className="text-slate-400 text-sm">Direct ingestion of MODIS (NDVI), SMAP (Surface Soil Moisture), and CHIRPS (Precipitation) data streams.</p>
              </div>
            </div>
            
            <div className="flex gap-4">
              <div className="flex-shrink-0 w-12 h-12 rounded-xl bg-slate-800 border border-slate-700 flex items-center justify-center">
                <BrainCircuit className="w-6 h-6 text-emerald-400" />
              </div>
              <div>
                <h4 className="text-xl font-bold mb-1">TensorFlow / Keras Backend</h4>
                <p className="text-slate-400 text-sm">Python and FastAPI drive the high-performance ML inference, executing deep learning algorithms instantly.</p>
              </div>
            </div>

            <div className="flex gap-4">
              <div className="flex-shrink-0 w-12 h-12 rounded-xl bg-slate-800 border border-slate-700 flex items-center justify-center">
                <Layers className="w-6 h-6 text-emerald-400" />
              </div>
              <div>
                <h4 className="text-xl font-bold mb-1">Modern Frontend Stack</h4>
                <p className="text-slate-400 text-sm">React 19, Vite, Tailwind CSS, and Recharts deliver a blazing-fast, interactive Command Center.</p>
              </div>
            </div>
          </div>
        </div>
        
        {/* Visual representation of the pipeline */}
        <div className="relative p-8 bg-slate-800/50 rounded-2xl border border-slate-700 backdrop-blur-sm">
          <div className="absolute -top-4 -right-4 bg-emerald-500 text-slate-900 text-xs font-bold px-3 py-1 rounded-full flex items-center gap-1 shadow-lg shadow-emerald-500/20">
            <CloudOff className="w-3 h-3" /> Cloud-Resistant
          </div>
          
          <div className="flex flex-col gap-4">
            {/* Input Layer */}
            <div className="flex justify-between items-center p-4 bg-slate-800 rounded-xl border border-slate-600">
              <span className="text-sm font-semibold text-slate-300">Data Sources</span>
              <div className="flex gap-2">
                <span className="px-2 py-1 text-[10px] bg-slate-700 rounded text-slate-300 font-mono">MODIS</span>
                <span className="px-2 py-1 text-[10px] bg-slate-700 rounded text-slate-300 font-mono">SMAP</span>
                <span className="px-2 py-1 text-[10px] bg-slate-700 rounded text-slate-300 font-mono">CHIRPS</span>
              </div>
            </div>
            
            <div className="flex justify-center"><ArrowRight className="w-5 h-5 text-slate-500 rotate-90" /></div>
            
            {/* Processing Layer */}
            <div className="p-4 bg-gradient-to-r from-emerald-900/40 to-teal-900/40 rounded-xl border border-emerald-500/30">
              <div className="flex items-center gap-3 mb-3">
                <BrainCircuit className="w-5 h-5 text-emerald-400" />
                <span className="text-sm font-bold text-white">CNN-LSTM Hybrid Model</span>
              </div>
              <div className="grid grid-cols-2 gap-2">
                <div className="bg-slate-900/50 p-2 rounded text-xs text-slate-300 border border-slate-700 text-center">CNN: Cross-Feature Extract</div>
                <div className="bg-slate-900/50 p-2 rounded text-xs text-slate-300 border border-slate-700 text-center">LSTM: Time Dependencies</div>
              </div>
            </div>

            <div className="flex justify-center"><ArrowRight className="w-5 h-5 text-slate-500 rotate-90" /></div>
            
            {/* Output Layer */}
            <div className="flex justify-between items-center p-4 bg-slate-800 rounded-xl border border-slate-600">
              <span className="text-sm font-semibold text-slate-300">Insights Output</span>
              <div className="flex gap-2">
                <span className="px-2 py-1 text-[10px] bg-emerald-500/20 text-emerald-400 rounded font-bold border border-emerald-500/30">3-Month Forecast</span>
                <span className="px-2 py-1 text-[10px] bg-emerald-500/20 text-emerald-400 rounded font-bold border border-emerald-500/30">Fraud Flagging</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </section>
);

const AlertsAndStats = () => (
  <section className="py-24 bg-white border-b border-slate-100">
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
      <div className="grid md:grid-cols-2 gap-16 items-center">
        {/* Live Alerts Visual */}
        <div className="order-2 md:order-1 relative">
          <div className="absolute inset-0 bg-slate-50 rounded-3xl transform -rotate-3 scale-105 z-0"></div>
          <div className="relative bg-white border border-slate-200 rounded-2xl shadow-xl p-6 z-10">
            <div className="flex items-center justify-between mb-6 pb-4 border-b border-slate-100">
              <h4 className="font-bold text-slate-800 flex items-center gap-2">
                <BellRing className="w-5 h-5 text-amber-500" />
                Live Alert System
              </h4>
              <span className="flex h-3 w-3 relative">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                <span className="relative inline-flex rounded-full h-3 w-3 bg-emerald-500"></span>
              </span>
            </div>
            
            <div className="space-y-4">
              <div className="p-4 rounded-lg bg-red-50 border border-red-100 flex gap-4 items-start">
                <div className="w-2 h-2 rounded-full bg-red-500 mt-1.5 flex-shrink-0"></div>
                <div>
                  <h5 className="font-bold text-sm text-slate-800 mb-1">Critical Moisture Drop: Anantapur</h5>
                  <p className="text-xs text-slate-600">SMI dropped below 0.15 threshold. Automated claim verification protocols tightened for this district.</p>
                  <span className="text-[10px] text-slate-400 font-mono mt-2 block">10 mins ago</span>
                </div>
              </div>
              <div className="p-4 rounded-lg bg-amber-50 border border-amber-100 flex gap-4 items-start">
                <div className="w-2 h-2 rounded-full bg-amber-500 mt-1.5 flex-shrink-0"></div>
                <div>
                  <h5 className="font-bold text-sm text-slate-800 mb-1">NDVI Divergence: Kurnool</h5>
                  <p className="text-xs text-slate-600">Recent claims show high yield loss, but satellite NDVI indicates stable vegetation. Flagging 12 claims for review.</p>
                  <span className="text-[10px] text-slate-400 font-mono mt-2 block">1 hour ago</span>
                </div>
              </div>
              <div className="p-4 rounded-lg bg-slate-50 border border-slate-100 flex gap-4 items-start opacity-70">
                <div className="w-2 h-2 rounded-full bg-emerald-500 mt-1.5 flex-shrink-0"></div>
                <div>
                  <h5 className="font-bold text-sm text-slate-800 mb-1">Forecast Update: Chittoor</h5>
                  <p className="text-xs text-slate-600">3-month NDVI prediction upgraded to 'Stable'. Risk profile adjusted accordingly.</p>
                  <span className="text-[10px] text-slate-400 font-mono mt-2 block">3 hours ago</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Stats Content */}
        <div className="order-1 md:order-2">
          <h2 className="text-3xl md:text-4xl font-extrabold text-slate-900 mb-6">Actionable Intelligence, Instantly.</h2>
          <p className="text-lg text-slate-600 mb-8 leading-relaxed">
            Stop waiting months for manual field surveys. ClaimGuard Sentinel notifies your risk teams instantly when a district's vegetation health drops below critical parameters, allowing for preemptive reserve allocation.
          </p>
          
          <div className="grid grid-cols-2 gap-6">
            <div>
              <div className="text-4xl font-black text-emerald-600 mb-2">3 Mo.</div>
              <div className="text-sm font-bold text-slate-800 mb-1">Forecast Horizon</div>
              <div className="text-xs text-slate-500">Predict drought conditions well before they strike.</div>
            </div>
            <div>
              <div className="text-4xl font-black text-emerald-600 mb-2">13</div>
              <div className="text-sm font-bold text-slate-800 mb-1">Districts Monitored</div>
              <div className="text-xs text-slate-500">Comprehensive coverage across drought-prone regions.</div>
            </div>
            <div>
              <div className="text-4xl font-black text-emerald-600 mb-2">6</div>
              <div className="text-sm font-bold text-slate-800 mb-1">Key Metrics Analyzed</div>
              <div className="text-xs text-slate-500">From SMI to NDVI standard deviation per district.</div>
            </div>
            <div>
              <div className="text-4xl font-black text-emerald-600 mb-2">&lt;1s</div>
              <div className="text-sm font-bold text-slate-800 mb-1">Claim Verification</div>
              <div className="text-xs text-slate-500">Reduced from months to fractions of a second.</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </section>
);

const CTA = ({ onExplore }) => (
  <section className="py-24 relative overflow-hidden bg-emerald-600">
    <div className="absolute inset-0 bg-[url('https://www.transparenttextures.com/patterns/carbon-fibre.png')] opacity-10"></div>
    <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10 text-center">
      <h2 className="text-4xl md:text-5xl font-extrabold text-white mb-6 tracking-tight">
        Ready to Transform Agricultural Insurance?
      </h2>
      <p className="text-xl text-emerald-100 mb-10">
        Empower your claims investigators and risk actuaries with military-grade satellite monitoring and deep learning forecasts.
      </p>
      <div className="flex flex-col sm:flex-row justify-center gap-4">
        <button onClick={onExplore} className="bg-slate-900 hover:bg-slate-800 text-white px-8 py-4 rounded-xl font-bold text-lg transition-all shadow-xl flex items-center justify-center gap-2">
          Launch Command Center
        </button>
        <a href="/Research_Paper.pdf" target="_blank" rel="noreferrer" className="bg-white/10 hover:bg-white/20 text-white border border-white/30 px-8 py-4 rounded-xl font-bold text-lg transition-all flex items-center justify-center gap-2">
          <FileText className="w-5 h-5" />
          Read Research Paper
        </a>
      </div>
    </div>
  </section>
);

const Footer = ({ onExplore }) => (
  <footer className="bg-slate-900 text-slate-400 py-12 border-t border-slate-800">
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
      <div className="flex flex-col md:flex-row justify-between items-center mb-8 pb-8 border-b border-slate-800 gap-6 text-center md:text-left">
        <div>
          <div className="flex items-center gap-2 mb-4 justify-center md:justify-start">
            <div className="bg-emerald-600 p-1.5 rounded-md">
              <Satellite className="w-5 h-5 text-white" />
            </div>
            <span className="font-bold text-xl text-white tracking-tight">ClaimGuard<span className="text-emerald-500">Sentinel</span></span>
          </div>
          <p className="text-sm max-w-sm mx-auto md:mx-0">
            AI-powered platform designed for agricultural insurers to combat fraudulent crop claims and enable proactive drought risk management.
          </p>
        </div>
        <div className="flex gap-8">
          <button onClick={onExplore} className="hover:text-emerald-400 transition-colors font-medium">Launch Platform</button>
          <a href="/Research_Paper.pdf" target="_blank" rel="noreferrer" className="hover:text-emerald-400 transition-colors font-medium">Research Paper</a>
        </div>
      </div>
      <div className="flex flex-col md:flex-row justify-between items-center text-xs">
        <p>&copy; {new Date().getFullYear()} ClaimGuard Sentinel. All rights reserved.</p>
        <div className="mt-4 md:mt-0 flex gap-4">
          <span>Google Earth Engine Powered</span>
          <span>•</span>
          <span>South India Coverage</span>
        </div>
      </div>
    </div>
  </footer>
);

// Helper Icon for Technology Section
const DatabaseIcon = (props) => (
  <svg
    {...props}
    xmlns="http://www.w3.org/2000/svg"
    width="24"
    height="24"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
  >
    <ellipse cx="12" cy="5" rx="9" ry="3" />
    <path d="M3 5V19A9 3 0 0 0 21 19V5" />
    <path d="M3 12A9 3 0 0 0 21 12" />
  </svg>
);

export default function ClaimGuardLandingPage({ onExplore }) {
  return (
    <div className="min-h-screen bg-slate-50 font-sans selection:bg-emerald-200 selection:text-emerald-900">
      <Navbar onExplore={onExplore} />
      <Hero onExplore={onExplore} />
      <ProblemContext />
      <Features />
      <Technology />
      <AlertsAndStats />
      <CTA onExplore={onExplore} />
      <Footer onExplore={onExplore} />
    </div>
  );
}
