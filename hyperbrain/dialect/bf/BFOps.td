// Copyright 2025 PragmaTwice
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef BF_OPS
#define BF_OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"

def BF_Dialect : Dialect {
  let name = "bf";
  let summary = "BrainFuck";
  let cppNamespace = "::hyperbrain::bf";

  let useDefaultTypePrinterParser = 1;
}

def BF_Ptr : TypeDef<BF_Dialect, "Ptr"> {
  let mnemonic = "ptr";
}

class BF_Op<string mnemonic, list<Trait> traits = []> :
    Op<BF_Dialect, mnemonic, traits> {
  let extraClassDeclaration = "";
}

def BF_NextOp : BF_Op<"next"> {
  let summary = "increase the pointer to point to the next byte";

  let arguments = (ins
    BF_Ptr:$p
  );

  let results = (outs
    BF_Ptr:$res
  );
  
  let assemblyFormat = [{
    $p attr-dict
  }];
}

def BF_PrevOp : BF_Op<"prev"> {
  let summary = "decrease the pointer to point to the previous byte";

  let arguments = (ins
    BF_Ptr:$p
  );

  let results = (outs
    BF_Ptr:$res
  );
  
  let assemblyFormat = [{
    $p attr-dict
  }];
}

def BF_IncOp : BF_Op<"inc"> {
  let summary = "increase the pointed byte";

  let arguments = (ins
    BF_Ptr:$p
  );
  
  let assemblyFormat = [{
    $p attr-dict
  }];
}

def BF_DecOp : BF_Op<"dec"> {
  let summary = "decrease the pointed byte";

  let arguments = (ins
    BF_Ptr:$p
  );
  
  let assemblyFormat = [{
    $p attr-dict
  }];
}

def BF_OutputOp : BF_Op<"output"> {
  let summary = "print the pointed byte";

  let arguments = (ins
    BF_Ptr:$p
  );
  
  let assemblyFormat = [{
    $p attr-dict
  }];
}

def BF_InputOp : BF_Op<"input"> {
  let summary = "accept a byte and store into the pointed memory";

  let arguments = (ins
    BF_Ptr:$p
  );
  
  let assemblyFormat = [{
    $p attr-dict
  }];
}

def BF_WhileOp : BF_Op<"while"> {
  let summary = "while loop with condition (*p != 0)";

  let arguments = (ins
    BF_Ptr:$p
  );

  let results = (outs
    BF_Ptr:$res
  );

  let regions = (region
    SizedRegion<1>:$comp
  );
  
  let assemblyFormat = [{
    $p $comp attr-dict
  }];
}

def BF_YieldOp : BF_Op<"yield", [Terminator]> {
  let summary = "return operator in while loop";

  let arguments = (ins
    BF_Ptr:$p
  );
  
  let assemblyFormat = [{
    $p attr-dict
  }];
}

def BF_MainOp : BF_Op<"main"> {
  let summary = "entrypoint function of program";

  let regions = (region
    SizedRegion<1>:$comp
  );

  let assemblyFormat = [{
    `(` `)` $comp attr-dict
  }];
}

#endif
