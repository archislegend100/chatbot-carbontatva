import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import './globals.css';

const inter = Inter({ subsets: ['latin'] });

export const metadata: Metadata = {
  title: 'CarbonTatva | Industrial Energy Copilot',
  description: 'Advanced RAG-powered assistant for industrial energy efficiency.',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className={`${inter.className} bg-white text-gray-900 antialiased selection:bg-emerald-500/30 selection:text-emerald-900`}>
        {children}
      </body>
    </html>
  );
}
