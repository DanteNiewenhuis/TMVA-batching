#include <vector>

#include "TFile.h"
#include "TTree.h"
#include "TCanvas.h"
#include "TFrame.h"
#include "TH1F.h"
#include "TBenchmark.h"
#include "TRandom.h"
#include "TSystem.h"

void write()
{

   TFile *f = TFile::Open("data/large_data.root", "RECREATE");

   if (!f)
   {
      return;
   }

   int f1;
   float f2;
   bool f3;

   // Create a TTree
   TTree *t = new TTree("test_tree", "Tree with vectors");
   t->Branch("f1", &f1);
   t->Branch("f2", &f2);
   t->Branch("f3", &f3);

   Int_t i_max = 10000000;
   Int_t value = 1;
   for (Int_t i = 1; i < i_max + 1; i++)
   {
      f1 = i;
      f2 = i + 0.1;

      if (i > (i_max / 2))
      {
         f3 = true;
      }
      else
      {
         f3 = false;
      }

      // for (Int_t j = 0; j < i; ++j)
      // {
      //    f1.emplace_back(value++);
      //    if (i < (i_max / 2))
      //    {
      //       f4.emplace_back(true);
      //    }
      //    else
      //    {
      //       f4.emplace_back(false);
      //    }
      // }
      t->Fill();
   }
   f->Write();

   delete f;
}

void create_root_data()
{
   write();
}