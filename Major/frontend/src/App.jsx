import { useState } from 'react'
import { Header } from "@/components/dashboard/Header"
import { RiskMap } from "@/components/dashboard/RiskMap"
import { StatsWidget } from "@/components/dashboard/StatsWidget"
import { LiveAlerts } from "@/components/dashboard/LiveAlerts"
import { InvestigationView } from "@/components/dashboard/InvestigationView"
import { Sidebar } from "@/components/layout/Sidebar"

// Mock Claims Queue
const CLAIM_QUEUE = [
  { id: 'CLM-8892', name: 'Rajesh Kumar', district: 'Anantapur', status: 'Pending', type: 'Drought', date: 'Jan 12, 2024', risk: 'High' },
  { id: 'CLM-8893', name: 'Amit Singh', district: 'Chitradurga', status: 'Pending', type: 'Drought', date: 'Jan 14, 2024', risk: 'Low' },
  { id: 'CLM-8894', name: 'Priya Gowda', district: 'Ballari', status: 'Pending', type: 'Pest', date: 'Jan 15, 2024', risk: 'Medium' },
  { id: 'CLM-8895', name: 'Suresh Reddy', district: 'Anantapur', status: 'Pending', type: 'Drought', date: 'Jan 16, 2024', risk: 'High' },
  { id: 'CLM-8896', name: 'Venkatesh', district: 'Chitradurga', status: 'Pending', type: 'Flood', date: 'Jan 18, 2024', risk: 'Low' },
]

function App() {
  const [simulatedDate, setSimulatedDate] = useState("2024-01-12")
  const [activePage, setActivePage] = useState('dashboard') // 'dashboard' | 'investigation' | 'analytics' | 'settings'
  const [selectedClaimIndex, setSelectedClaimIndex] = useState(0)
  const [isCollapsed, setIsCollapsed] = useState(false)

  // Navigation Logic
  const handleNextClaim = () => {
    if (selectedClaimIndex < CLAIM_QUEUE.length - 1) {
      setSelectedClaimIndex(prev => prev + 1)
    } else {
      window.alert("Queue Completed for Today!")
      setActivePage('dashboard')
      setSelectedClaimIndex(0)
    }
  }

  const handleDistrictSelect = (district) => {
    // Find the first claim for this district to start investigation
    const index = CLAIM_QUEUE.findIndex(c => c.district === district.name)
    if (index !== -1) {
      setSelectedClaimIndex(index)
      setActivePage('investigation')
    } else {
      // Fallback if no specific claim mockup exists
      setActivePage('investigation')
    }
  }

  const currentClaim = CLAIM_QUEUE[selectedClaimIndex]

  return (
    <div className="flex h-screen bg-background text-foreground overflow-hidden font-sans">
      {/* Sidebar Navigation */}
      <Sidebar
        activePage={activePage}
        setActivePage={setActivePage}
        isCollapsed={isCollapsed}
        setIsCollapsed={setIsCollapsed}
      />

      <div className="flex-1 flex flex-col overflow-hidden">
        <Header simulatedDate={simulatedDate} setSimulatedDate={setSimulatedDate} />

        <main className="flex-1 overflow-hidden p-4 gap-4 relative">
          {activePage === 'dashboard' && (
            <div className="flex h-full gap-4 animate-in fade-in duration-300">
              <div className="flex-[2.5] rounded-xl border border-border/50 overflow-hidden shadow-2xl relative">
                <RiskMap onSelectDistrict={handleDistrictSelect} />
              </div>

              <div className="flex-1 flex flex-col gap-4 min-w-[350px]">
                <div className="h-[30%]">
                  <StatsWidget />
                </div>
                <div className="h-[70%] rounded-xl border border-border/50 bg-card shadow-lg overflow-hidden">
                  <LiveAlerts />
                </div>
              </div>
            </div>
          )}

          {activePage === 'investigation' && (
            <div className="w-full h-full animate-in slide-in-from-right-4 duration-300">
              <InvestigationView
                claim={currentClaim}
                onBack={() => setActivePage('dashboard')}
                onProcessClaim={handleNextClaim}
                queueLength={CLAIM_QUEUE.length}
                currentIndex={selectedClaimIndex}
              />
            </div>
          )}

          {/* Pages like Analytics/Settings would go here */}
          {(activePage !== 'dashboard' && activePage !== 'investigation') && (
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
