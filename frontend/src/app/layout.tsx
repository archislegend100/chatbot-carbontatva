import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'Industrial Energy Efficiency Copilot',
  description:
    'Advanced RAG-powered assistant for industrial energy efficiency — grounded in BEE thermal and electrical utility manuals. Ask technical questions, explore opportunities, troubleshoot systems, and generate checklists.',
  keywords: [
    'energy efficiency',
    'industrial',
    'BEE',
    'thermal utilities',
    'electrical utilities',
    'RAG chatbot',
    'energy audit',
  ],
  authors: [{ name: 'CarbonTatva' }],
  robots: 'noindex', // Internal prototype
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <head>
        <link rel="icon" href="/favicon.ico" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
      </head>
      <body>{children}</body>
    </html>
  );
}
