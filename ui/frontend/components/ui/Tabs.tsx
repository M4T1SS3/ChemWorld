'use client'

import { createContext, useContext, useState } from 'react'

interface TabsContextType {
  activeTab: string
  setActiveTab: (tab: string) => void
}

const TabsContext = createContext<TabsContextType | undefined>(undefined)

export function Tabs({
  defaultValue,
  children,
  className = '',
}: {
  defaultValue: string
  children: React.ReactNode
  className?: string
}) {
  const [activeTab, setActiveTab] = useState(defaultValue)

  return (
    <TabsContext.Provider value={{ activeTab, setActiveTab }}>
      <div className={className}>{children}</div>
    </TabsContext.Provider>
  )
}

export function TabsList({
  children,
  className = '',
}: {
  children: React.ReactNode
  className?: string
}) {
  return (
    <div className={`flex border-b border-border ${className}`}>
      {children}
    </div>
  )
}

export function TabsTrigger({
  value,
  children,
}: {
  value: string
  children: React.ReactNode
}) {
  const context = useContext(TabsContext)
  if (!context) throw new Error('TabsTrigger must be used within Tabs')

  const { activeTab, setActiveTab } = context
  const isActive = activeTab === value

  return (
    <button
      onClick={() => setActiveTab(value)}
      className={`px-6 py-3 text-sm font-medium transition-colors ${
        isActive
          ? 'text-primary border-b-2 border-primary'
          : 'text-text-muted hover:text-text'
      }`}
    >
      {children}
    </button>
  )
}

export function TabsContent({
  value,
  children,
  className = '',
}: {
  value: string
  children: React.ReactNode
  className?: string
}) {
  const context = useContext(TabsContext)
  if (!context) throw new Error('TabsContent must be used within Tabs')

  const { activeTab } = context
  if (activeTab !== value) return null

  return <div className={`pt-6 ${className}`}>{children}</div>
}
