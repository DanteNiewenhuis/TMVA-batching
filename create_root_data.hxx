#include <vector>

#include "TFile.h"
#include "TTree.h"
#include "TCanvas.h"
#include "TFrame.h"
#include "TH1F.h"
#include "TBenchmark.h"
#include "TRandom.h"
#include "TSystem.h"

void writeAllVectorTypes()
{

   TFile *f = TFile::Open("data/hvector.root", "RECREATE");

   if (!f)
   {
      return;
   }

   std::vector<bool> fB;
   std::vector<double> fD;
   std::vector<float> fF;
   std::vector<int> fI;
   std::vector<long> fL;
   std::vector<long long> fLL;
   std::vector<unsigned> fU;
   std::vector<unsigned long> fUL;
   std::vector<unsigned long long> fULL;

   // Create a TTree
   TTree *t = new TTree("test_tree", "Tree with vectors");
   t->Branch("fB", &fB);
   t->Branch("fD", &fD);
   t->Branch("fF", &fF);
   t->Branch("fI", &fI);
   t->Branch("fL", &fL);
   t->Branch("fLL", &fLL);
   t->Branch("fU", &fU);
   t->Branch("fUL", &fUL);
   t->Branch("fULL", &fULL);

   Int_t i_max = 100;
   for (Int_t i = 1; i <= i_max; i++)
   {

      fB.clear();
      fD.clear();
      fF.clear();
      fI.clear();
      fL.clear();
      fLL.clear();
      fU.clear();
      fUL.clear();
      fULL.clear();

      for (Int_t j = 0; j < i; j++)
      {
         fD.emplace_back(j);
         fF.emplace_back(j);
         fI.emplace_back(j);
         fL.emplace_back(j);
         fLL.emplace_back(j);
         fU.emplace_back(j);
         fUL.emplace_back(j);
         fULL.emplace_back(j);

         if (i < (i_max / 2))
         {
            fB.emplace_back(true);
         }
         else
         {
            fB.emplace_back(false);
         }
      }
      t->Fill();
   }
   f->Write();

   delete f;
}

void writeAllSingleTypes()
{
   TFile *f = TFile::Open("data/all_types.root", "RECREATE");

   // Create a TTree
   TTree *t = new TTree("test_tree", "Tree with vectors");

   short fS;
   unsigned short fs;
   int fI;
   unsigned int fi;
   float fF;
   float ff;
   double fD;
   double fd;
   long fL;
   unsigned long fl;
   long long fG;
   unsigned long long fg;
   bool fb;

   t->Branch("fS", &fS, "fS/S");
   t->Branch("fs", &fs, "fs/s");
   t->Branch("fI", &fI, "fI/I");
   t->Branch("fi", &fi, "fi/i");
   t->Branch("fF", &fF, "fF/F");
   t->Branch("ff", &ff, "ff/f");
   t->Branch("fD", &fD, "fD/D");
   t->Branch("fd", &fd, "fd/d");
   t->Branch("fL", &fL, "fL/L");
   t->Branch("fl", &fl, "fl/l");
   t->Branch("fG", &fG, "fG/G");
   t->Branch("fg", &fg, "fg/g");
   t->Branch("fb", &fb, "fb/O");

   int i_max = 10;
   for (Int_t i = 1; i <= i_max; i++)
   {
      fS = i;
      fs = i;
      fI = i;
      fi = i;
      fF = i;
      ff = i;
      fD = i;
      fd = i;
      fL = i;
      fl = i;
      fG = i;
      fg = i;
      fb = i;

      t->Fill();
   }

   f->Write();

   delete f;
}

void create_root_data()
{
   writeAllSingleTypes();
   writeAllVectorTypes();
}
