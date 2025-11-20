import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'ChemJEPA - Molecular Discovery',
  description: 'Hierarchical Latent World Models for Molecular Discovery',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
}
