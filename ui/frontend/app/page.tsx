'use client'

import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/Tabs'
import AnalyzeTab from '@/components/tabs/AnalyzeTab'
import DiscoverTab from '@/components/tabs/DiscoverTab'
import CompareTab from '@/components/tabs/CompareTab'

export default function Home() {
  return (
    <main className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b border-border bg-white">
        <div className="container mx-auto px-6 py-4">
          <h1 className="text-2xl font-semibold text-text">ChemJEPA</h1>
          <p className="text-sm text-text-muted mt-1">
            Hierarchical Latent World Models for Molecular Discovery
          </p>
        </div>
      </header>

      {/* Main Content */}
      <div className="container mx-auto px-6 py-8">
        <Tabs defaultValue="analyze">
          <TabsList>
            <TabsTrigger value="analyze">Analyze</TabsTrigger>
            <TabsTrigger value="discover">Discover</TabsTrigger>
            <TabsTrigger value="compare">Compare</TabsTrigger>
          </TabsList>

          <TabsContent value="analyze">
            <AnalyzeTab />
          </TabsContent>

          <TabsContent value="discover">
            <DiscoverTab />
          </TabsContent>

          <TabsContent value="compare">
            <CompareTab />
          </TabsContent>
        </Tabs>
      </div>
    </main>
  )
}
