import '../styles/globals.css';
import type { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'RoadSense India — Plate recognition & road inspection',
  description: 'Local, review-first number plate recognition and road inspection prototype.',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className="bg-command-bg text-command-text min-h-screen font-sans antialiased selection:bg-brand selection:text-command-bg">
        {children}
      </body>
    </html>
  );
}
