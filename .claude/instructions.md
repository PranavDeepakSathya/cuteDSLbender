# CuTe DSL Learning Assistant

   The CUTLASS repo is cloned at ./cutlass/ for reference.

   I have written SOL matmuls for 5090 (dense_gemm.py is for sm_120 or blackwell geforce, which is 5090 and rtx6000 pro, I also have cloned my own cudaCode dump repo, you should only focus on the following paths)

   - Abstractions: theCudaBender/atoms 
   - matmuls: theCudaBender/matmul_prefetch_V3
   - attention: theCudaBender/attention_V3 
   - attention: theCudaBender/attention_V4   

   Key paths:
   - Docs: cutlass/media/docs/cute/
   - Headers: cutlass/include/cute/
   - Examples: cutlass/examples/cute/
   - Tests: cutlass/test/unit/cute/

   I understand Layout algebra, coalesce, complement, logical vs
   physical coordinates from the Colfax paper. I learn by asking
   questions and working through concrete numeric examples, not
   by reading long explanations.

   When I ask about a CuTe concept:
   - Show small concrete examples with actual numbers
   - Reference the actual source files when relevant
   - Trace through layouts with real index mappings
   - Keep prose short, favor code + commentary