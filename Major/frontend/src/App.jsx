import { useState, useEffect } from 'react'
import { Header } from "@/components/dashboard/Header"
import { RiskMap } from "@/components/dashboard/RiskMap"
import { StatsWidget } from "@/components/dashboard/StatsWidget"
import { LiveAlerts } from "@/components/dashboard/LiveAlerts"
import { InvestigationView } from "@/components/dashboard/InvestigationView"
import { ForecastView } from "@/components/dashboard/ForecastView"
import { ClusterView } from "@/components/dashboard/ClusterView"
import { Sidebar } from "@/components/layout/Sidebar"
import { Toaster, toast } from 'sonner'
import { fetchDistricts, fetchAlerts } from "@/lib/api"

function App() {
  const [simulatedDate, setSimulatedDate] = useState("2026-02-01")
  const [activePage, setActivePage] = useState('dashboard')
  const [selectedClaimIndex, setSelectedClaimIndex] = useState(0)
  const [isCollapsed, setIsCollapsed] = useState(false)

  // Real data from API
  const [districts, setDistricts] = useState([])
  const [alerts, setAlerts] = useState([])
  const [loading, setLoading] = useState(true)
  const [apiError, setApiError] = useState(null)

  // Build claim queue from high-risk districts
  const claimQueue = districts
    .filter(d => d.risk > 0.5)
    .map((d, i) => ({
      id: `CLM-${9000 + i}`,
      name: `Farmer, ${d.name}`,
      district: d.name,
      status: 'Pending',
      type: 'Drought',
      date: d.latest_date,
      risk: d.risk > 0.7 ? 'High' : 'Medium',
      ndvi: d.ndvi,
      smi: d.smi,
      rainfall: d.rainfall,
    }))

  // Fetch real data from backend
  useEffect(() => {
    async function loadData() {
      setLoading(true)
      setApiError(null)
      try {
        const [districtData, alertData] = await Promise.all([
          fetchDistricts(),
          fetchAlerts(),
        ])
        setDistricts(districtData)
        setAlerts(alertData)
      } catch (err) {
        console.error("Failed to load data:", err)
        setApiError(err.message)
        toast.error("Backend connection failed", {
          description: "Make sure the FastAPI server is running on port 8000"
        })
      } finally {
        setLoading(false)
      }
    }
    loadData()
  }, [])

  // Navigation Logic
  const handleNextClaim = () => {
    toast.success("Claim processed successfully")
    if (selectedClaimIndex < claimQueue.length - 1) {
      setSelectedClaimIndex(prev => prev + 1)
    } else {
      toast.success("Queue Completed for Today!", {
        description: "All pending claims have been processed."
      })
      setActivePage('dashboard')
      setSelectedClaimIndex(0)
    }
  }

  const handleDistrictSelect = (district) => {
    const index = claimQueue.findIndex(c => c.district === district.name)
    if (index !== -1) {
      setSelectedClaimIndex(index)
      setActivePage('investigation')
      toast.info(`Viewing claims for ${district.name}`)
    } else {
      // Even if no claim exists, enter investigation mode with this district's data
      setActivePage('investigation')
      toast.info(`Entered investigation mode for ${district.name}`)
    }
  }

  const currentClaim = claimQueue[selectedClaimIndex] || null

  return (
    <div className="flex h-screen bg-background text-foreground overflow-hidden font-sans">
      <Toaster position="top-center" richColors />
      <Sidebar
        activePage={activePage}
        setActivePage={setActivePage}
        isCollapsed={isCollapsed}
        setIsCollapsed={setIsCollapsed}
        claimCount={claimQueue.length}
      />

      <div className="flex-1 flex flex-col overflow-hidden">
        <Header simulatedDate={simulatedDate} setSimulatedDate={setSimulatedDate} apiStatus={!apiError} />

        <main className="flex-1 overflow-hidden p-4 gap-4 relative">
          {loading && (
            <div className="h-full flex items-center justify-center flex-col gap-3">
              <div className="h-8 w-8 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
              <p className="text-sm text-muted-foreground">Connecting to ClaimGuard API...</p>
            </div>
          )}

          {!loading && activePage === 'dashboard' && (
            <div className="flex h-full gap-4 animate-in fade-in duration-300">
              <div className="flex-[2.5] rounded-xl border border-border/50 overflow-hidden shadow-2xl relative">
                <RiskMap districts={districts} onSelectDistrict={handleDistrictSelect} />
              </div>

              <div className="flex-1 flex flex-col gap-3 min-w-[350px]">
                <div className="shrink-0">
                  <StatsWidget districts={districts} />
                </div>
                <div className="flex-1 rounded-xl border border-border/50 bg-card shadow-lg overflow-hidden min-h-0">
                  <LiveAlerts alerts={alerts} />
                </div>
              </div>
            </div>
          )}

          {!loading && activePage === 'investigation' && (
            <div className="w-full h-full animate-in slide-in-from-right-4 duration-300">
              <InvestigationView
                claim={currentClaim}
                onBack={() => setActivePage('dashboard')}
                onProcessClaim={handleNextClaim}
                queueLength={claimQueue.length}
                currentIndex={selectedClaimIndex}
              />
            </div>
          )}

          {!loading && activePage === 'forecast' && (
            <div className="w-full h-full animate-in fade-in duration-300">
              <ForecastView />
            </div>
          )}

          {!loading && activePage === 'analytics' && (
            <div className="w-full h-full animate-in fade-in duration-300">
              <ClusterView />
            </div>
          )}

          {!loading && !['dashboard', 'investigation', 'forecast', 'analytics'].includes(activePage) && (
            <div className="h-full flex items-center justify-center text-muted-foreground flex-col gap-2">
              <div className="h-12 w-12 rounded-full border-2 border-dashed border-muted-foreground/30 flex items-center justify-center">
                🚧
              </div>
              <p>Module Under Construction</p>
            </div>
          )}
        </main>
      </div>
    </div>
  )
}

export default App
