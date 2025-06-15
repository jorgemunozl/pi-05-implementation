
import React from 'react';
import Navigation from '@/components/Navigation';
import Hero from '@/components/Hero';
import FeaturedSection from '@/components/FeaturedSection';
import ExperimentsSection from '@/components/ExperimentsSection';
import LatexSection from '@/components/LatexSection';
import MathSection from '@/components/MathSection';
import ComputerScienceSection from '@/components/ComputerScienceSection';
import Footer from '@/components/Footer';

const Index = () => {
  return (
    <div className="min-h-screen bg-white">
      <Navigation />
      <Hero />
      <FeaturedSection />
      <ExperimentsSection />
      <LatexSection />
      <MathSection />
      <ComputerScienceSection />
      <Footer />
    </div>
  );
};

export default Index;
