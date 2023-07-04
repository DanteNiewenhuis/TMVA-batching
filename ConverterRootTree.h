#include "TTree.h"
#include "TFile.h"

// struct to create ROOT Tree from arrays obtained from
// an h5 file
struct ConverterRootTree {

   int blocksize = 0;
   std::vector<float> fScalars;
   std::vector<std::vector<float>> fVecs;
   std::vector<double> fScalarBuffer;
   std::vector<float> fVecBuffer;
   TFile * file = nullptr;
   TTree * tree = nullptr;

   ConverterRootTree(std::string filename, std::string treename, int n, int ns, int nv) {
      file = TFile::Open(filename.c_str(),"RECREATE");
      tree = new TTree(treename.c_str(),treename.c_str());
      blocksize = n;
      fScalars.reserve(ns);
      fVecs.reserve(nv);
   }
   void Write() {
      tree->Write();
      file->Close();
      delete file;
   }

   void AddFloatBranch(std::string name){
      std::string type = name + "/F";
      fScalars.push_back(0);
      tree->Branch(name.c_str(),&fScalars[fScalars.size()-1],type.c_str());
      fScalarBuffer.resize(fScalars.size()*blocksize);
   }
   void AddVecBranch(std::string name, int size){
      fVecs.push_back(std::vector<float>(size));
      tree->Branch(name.c_str(),&fVecs[fVecs.size()-1]);
      fVecBuffer.resize(fVecs.size()*size*blocksize);
   }

   void SetScalarData(int n, const double * data) {
      std::copy(data, data+n, fScalarBuffer.begin());
   }
   void SetVecData(int n, const float * data) {
      std::copy(data,data+n,fVecBuffer.begin());

   }
   void Fill() {
      int ns = fScalars.size();
      int nv = fVecs.size();
      for (int i = 0; i < blocksize; i++) {
         //std::cout << "copying event " << i << std::endl;
         for (int j = 0; j < ns; j++) {
            fScalars[j] = fScalarBuffer[i*ns+j];
         }
         for (int j = 0; j < nv; j++) {
            // fVecBuffer is data of shape (nevts, nvar, vsize)
            int first = i*nv*fVecs[j].size() + j*fVecs[j].size();
            int last = first + fVecs[j].size();
            //std::cout << j << " filling buffer from " << first << "   " << last << std::endl;
            std::copy(fVecBuffer.begin()+first,fVecBuffer.begin()+last ,fVecs[j].begin());
         }
         //std::cout << "filling ttree " << std::endl;
         tree->Fill();
      }
   }
};