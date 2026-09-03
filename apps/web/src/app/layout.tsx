import '../styles/globals.css';
import type { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'RoadSense India - Dashcam Operations Command Center',
  description: 'Industrial-grade local dashcam pothole inspection, verification, and field dossier platform.',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className="bg-command-bg text-command-text min-h-screen font-sans antialiased selection:bg-radar-bright selection:text-command-bg">
        {children}
      </body>
    </html>
  );
}
