#!/usr/bin/env python3
"""
重新生成100样本评估集 - 确保训练集和评估集反应类型一致
"""

import json
import random
from pathlib import Path

random.seed(42)

OUTPUT_DIR = Path("./data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 使用扩展版的真实化学反应数据集（从prepare_data_expanded.py复制）
REAL_REACTIONS = [
    # ============================================================
    # 酯化反应 (Esterification) - Easy
    # ============================================================
    {"type": "esterification", "reactants_smiles": ["CC(=O)O", "CCO"], "product_smiles": "CC(=O)OCC",
     "conditions": "H2SO4 cat., 80°C, 2h", "difficulty": "easy"},
    {"type": "esterification", "reactants_smiles": ["CCCC(=O)O", "CC(C)O"], "product_smiles": "CCCC(=O)OC(C)C",
     "conditions": "H2SO4 cat., reflux, 3h", "difficulty": "easy"},
    {"type": "esterification", "reactants_smiles": ["OC(=O)c1ccccc1", "CCO"], "product_smiles": "CCOC(=O)c1ccccc1",
     "conditions": "H2SO4 cat., reflux, 4h", "difficulty": "easy"},
    {"type": "esterification", "reactants_smiles": ["CC(=O)O", "CCCCO"], "product_smiles": "CCCCOC(=O)C",
     "conditions": "H2SO4 cat., reflux, 2h", "difficulty": "easy"},
    {"type": "esterification", "reactants_smiles": ["CC(=O)O", "CC(C)O"], "product_smiles": "CC(=O)OC(C)C",
     "conditions": "H2SO4 cat., 85°C, 3h", "difficulty": "easy"},
    {"type": "esterification", "reactants_smiles": ["CCCCC(=O)O", "CCO"], "product_smiles": "CCCCC(=O)OCC",
     "conditions": "H2SO4 cat., reflux, 2h", "difficulty": "easy"},
    {"type": "esterification", "reactants_smiles": ["CC(=O)O", "CCCCCO"], "product_smiles": "CCCCCOC(=O)C",
     "conditions": "H2SO4 cat., reflux, 3h", "difficulty": "easy"},

    # ============================================================
    # 酯化反应 (Esterification) - Medium
    # ============================================================
    {"type": "esterification", "reactants_smiles": ["OC(=O)C1=CC=CC=C1O", "CC(=O)OC(=O)C"], "product_smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",
     "conditions": "H2SO4 cat., 85°C, 15min", "difficulty": "medium"},
    {"type": "esterification", "reactants_smiles": ["OC(=O)c1cccc(C(=O)O)c1", "CCO"], "product_smiles": "CCOC(=O)c1cccc(C(=O)O)c1",
     "conditions": "H2SO4 cat., reflux, 6h", "difficulty": "medium"},
    {"type": "esterification", "reactants_smiles": ["OC(=O)CC(=O)O", "CCO"], "product_smiles": "CCOC(=O)CC(=O)O",
     "conditions": "H2SO4 cat., reflux, 4h", "difficulty": "medium"},
    {"type": "esterification", "reactants_smiles": ["OC(=O)c1ccccc1", "CC(C)O"], "product_smiles": "CC(C)OC(=O)c1ccccc1",
     "conditions": "H2SO4 cat., reflux, 5h", "difficulty": "medium"},

    # ============================================================
    # 还原反应 (Reduction) - Easy
    # ============================================================
    {"type": "reduction", "reactants_smiles": ["CC(=O)C"], "product_smiles": "CC(C)O",
     "conditions": "NaBH4, MeOH, 0°C", "difficulty": "easy"},
    {"type": "reduction", "reactants_smiles": ["O=CC1=CC=CC=C1"], "product_smiles": "OCC1=CC=CC=C1",
     "conditions": "NaBH4, EtOH, rt", "difficulty": "easy"},
    {"type": "reduction", "reactants_smiles": ["O=C1CCCCC1"], "product_smiles": "OC1CCCCC1",
     "conditions": "NaBH4, MeOH, 0°C", "difficulty": "easy"},
    {"type": "reduction", "reactants_smiles": ["CC(=O)c1ccccc1"], "product_smiles": "CC(O)c1ccccc1",
     "conditions": "NaBH4, EtOH, 0°C", "difficulty": "easy"},
    {"type": "reduction", "reactants_smiles": ["CC=O"], "product_smiles": "CCO",
     "conditions": "NaBH4, MeOH, 0°C", "difficulty": "easy"},
    {"type": "reduction", "reactants_smiles": ["O=CCC"], "product_smiles": "OCCCC",
     "conditions": "NaBH4, EtOH, rt", "difficulty": "easy"},
    {"type": "reduction", "reactants_smiles": ["O=Cc1ccccc1"], "product_smiles": "OCc1ccccc1",
     "conditions": "NaBH4, MeOH, 0°C", "difficulty": "easy"},

    # ============================================================
    # 还原反应 (Reduction) - Medium
    # ============================================================
    {"type": "reduction", "reactants_smiles": ["O=C(O)C1=CC=CC=C1"], "product_smiles": "OCC1=CC=CC=C1",
     "conditions": "LiAlH4, THF, 0°C→rt", "difficulty": "medium"},
    {"type": "reduction", "reactants_smiles": ["CC=CC=O"], "product_smiles": "CC=CCO",
     "conditions": "NaBH4, CeCl3, MeOH", "difficulty": "medium"},
    {"type": "reduction", "reactants_smiles": ["O=C(O)c1ccccc1"], "product_smiles": "OCc1ccccc1",
     "conditions": "LiAlH4, THF, reflux", "difficulty": "medium"},
    {"type": "reduction", "reactants_smiles": ["CC(=O)OCC"], "product_smiles": "CCO",
     "conditions": "LiAlH4, Et2O, reflux", "difficulty": "medium"},
    {"type": "reduction", "reactants_smiles": ["O=C(O)CC(=O)O"], "product_smiles": "OCC(O)CCO",
     "conditions": "LiAlH4, THF, reflux", "difficulty": "medium"},

    # ============================================================
    # 氧化反应 (Oxidation) - Easy
    # ============================================================
    {"type": "oxidation", "reactants_smiles": ["OCC1=CC=CC=C1"], "product_smiles": "O=CC1=CC=CC=C1",
     "conditions": "PCC, DCM, rt", "difficulty": "easy"},
    {"type": "oxidation", "reactants_smiles": ["CC(C)O"], "product_smiles": "CC(=O)C",
     "conditions": "Jones reagent, acetone, 0°C", "difficulty": "easy"},
    {"type": "oxidation", "reactants_smiles": ["OC1CCCCC1"], "product_smiles": "O=C1CCCCC1",
     "conditions": "PCC, DCM, rt", "difficulty": "easy"},
    {"type": "oxidation", "reactants_smiles": ["CC(O)c1ccccc1"], "product_smiles": "CC(=O)c1ccccc1",
     "conditions": "PCC, DCM, rt", "difficulty": "easy"},
    {"type": "oxidation", "reactants_smiles": ["CCO"], "product_smiles": "CC=O",
     "conditions": "PCC, DCM, rt", "difficulty": "easy"},
    {"type": "oxidation", "reactants_smiles": ["OCCc1ccccc1"], "product_smiles": "O=CCc1ccccc1",
     "conditions": "PCC, DCM, rt", "difficulty": "easy"},
    {"type": "oxidation", "reactants_smiles": ["OCc1ccccc1"], "product_smiles": "O=Cc1ccccc1",
     "conditions": "PCC, DCM, rt", "difficulty": "easy"},

    # ============================================================
    # 氧化反应 (Oxidation) - Medium
    # ============================================================
    {"type": "oxidation", "reactants_smiles": ["OCC1=CC=CC=C1"], "product_smiles": "OC(=O)C1=CC=CC=C1",
     "conditions": "KMnO4, NaOH, H2O, 60°C", "difficulty": "medium"},
    {"type": "oxidation", "reactants_smiles": ["CCc1ccccc1"], "product_smiles": "OC(=O)c1ccccc1",
     "conditions": "KMnO4, NaOH, H2O, reflux", "difficulty": "medium"},
    {"type": "oxidation", "reactants_smiles": ["CC(C)c1ccccc1"], "product_smiles": "OC(=O)c1ccccc1",
     "conditions": "KMnO4, NaOH, H2O, reflux", "difficulty": "medium"},
    {"type": "oxidation", "reactants_smiles": ["CC(C)(C)O"], "product_smiles": "CC(=O)C",
     "conditions": "H2CrO4, H2O, acetone", "difficulty": "medium"},

    # ============================================================
    # 亲核取代 (SN2) - Easy
    # ============================================================
    {"type": "substitution_SN2", "reactants_smiles": ["CCBr", "[OH-]"], "product_smiles": "CCO",
     "conditions": "NaOH, H2O, 60°C", "difficulty": "easy"},
    {"type": "substitution_SN2", "reactants_smiles": ["CCCl", "[NH3]"], "product_smiles": "CCN",
     "conditions": "NH3 (excess), EtOH, sealed tube, 100°C", "difficulty": "easy"},
    {"type": "substitution_SN2", "reactants_smiles": ["CCBr", "CCN"], "product_smiles": "CCNCC",
     "conditions": "Et3N, DMF, 80°C", "difficulty": "easy"},
    {"type": "substitution_SN2", "reactants_smiles": ["CCBr", "CCO"], "product_smiles": "CCOCC",
     "conditions": "NaH, DMF, 0°C", "difficulty": "easy"},
    {"type": "substitution_SN2", "reactants_smiles": ["CCBr", "CC#N"], "product_smiles": "CCC#N",
     "conditions": "NaCN, DMF, 60°C", "difficulty": "easy"},

    # ============================================================
    # 亲核取代 (SN2) - Medium
    # ============================================================
    {"type": "substitution_SN2", "reactants_smiles": ["BrCCBr", "[S-]"], "product_smiles": "SCCBr",
     "conditions": "NaSH, DMF, 0°C", "difficulty": "medium"},
    {"type": "substitution_SN2", "reactants_smiles": ["CI", "[N-]#[C-]"], "product_smiles": "CC#N",
     "conditions": "NaCN, DMF, 60°C", "difficulty": "medium"},
    {"type": "substitution_SN2", "reactants_smiles": ["CCBr", "CCS[Na]"], "product_smiles": "CCSCC",
     "conditions": "NaSEt, DMF, rt", "difficulty": "medium"},
    {"type": "substitution_SN2", "reactants_smiles": ["CCCl", "c1ccncc1"], "product_smiles": "CCn1ccncc1",
     "conditions": "K2CO3, DMF, 100°C", "difficulty": "medium"},

    # ============================================================
    # 消除反应 (Elimination) - Easy
    # ============================================================
    {"type": "elimination", "reactants_smiles": ["CC(O)C"], "product_smiles": "CC=C",
     "conditions": "H2SO4 conc., 170°C", "difficulty": "easy"},
    {"type": "elimination", "reactants_smiles": ["CC(O)(C)C"], "product_smiles": "CC(C)=C",
     "conditions": "H2SO4 conc., 85°C", "difficulty": "easy"},
    {"type": "elimination", "reactants_smiles": ["CC(Br)C"], "product_smiles": "CC=C",
     "conditions": "KOH, EtOH, reflux", "difficulty": "easy"},

    # ============================================================
    # 消除反应 (Elimination) - Medium
    # ============================================================
    {"type": "elimination", "reactants_smiles": ["CC(Br)CC"], "product_smiles": "CC=CC",
     "conditions": "KOH, EtOH, reflux", "difficulty": "medium"},
    {"type": "elimination", "reactants_smiles": ["OC1CCCCC1"], "product_smiles": "C=C1CCCCC1",
     "conditions": "H2SO4 conc., 165°C", "difficulty": "medium"},
    {"type": "elimination", "reactants_smiles": ["CC(O)(CC)CCC"], "product_smiles": "CC(=C)CCC",
     "conditions": "H2SO4 conc., 100°C", "difficulty": "medium"},
    {"type": "elimination", "reactants_smiles": ["CC(Br)(CC)CCC"], "product_smiles": "CC(=C)CCC",
     "conditions": "KOH, EtOH, reflux", "difficulty": "medium"},

    # ============================================================
    # 加成反应 (Addition) - Easy
    # ============================================================
    {"type": "addition", "reactants_smiles": ["CC=C", "BrBr"], "product_smiles": "CC(Br)CBr",
     "conditions": "DCM, 0°C, dark", "difficulty": "easy"},
    {"type": "addition", "reactants_smiles": ["CC=C", "O"], "product_smiles": "CC(O)C",
     "conditions": "H2SO4 cat., H2O, 80°C", "difficulty": "easy"},
    {"type": "addition", "reactants_smiles": ["CC=C", "[H][H]"], "product_smiles": "CCC",
     "conditions": "Pd/C, H2, 1 atm, rt", "difficulty": "easy"},
    {"type": "addition", "reactants_smiles": ["CC=C", "Cl"], "product_smiles": "CC(Cl)C",
     "conditions": "HCl gas, 0°C", "difficulty": "easy"},
    {"type": "addition", "reactants_smiles": ["CC=CC", "BrBr"], "product_smiles": "CC(Br)C(Br)C",
     "conditions": "CCl4, 0°C", "difficulty": "easy"},
    {"type": "addition", "reactants_smiles": ["c1ccccc1", "[H][H]"], "product_smiles": "C1CCCCC1",
     "conditions": "Pt, H2, 100 atm, 200°C", "difficulty": "easy"},

    # ============================================================
    # 加成反应 (Addition) - Medium
    # ============================================================
    {"type": "addition", "reactants_smiles": ["c1ccccc1", "ClCl"], "product_smiles": "Clc1ccccc1",
     "conditions": "FeCl3 cat., 25°C, dark", "difficulty": "medium"},
    {"type": "addition", "reactants_smiles": ["CC=CC", "ClCl"], "product_smiles": "CC(Cl)C(Cl)C",
     "conditions": "CCl4, 0°C", "difficulty": "medium"},
    {"type": "addition", "reactants_smiles": ["C=Cc1ccccc1", "BrBr"], "product_smiles": "BrCC(Br)c1ccccc1",
     "conditions": "DCM, 0°C", "difficulty": "medium"},
    {"type": "addition", "reactants_smiles": ["C=Cc1ccccc1", "[H][H]"], "product_smiles": "CCc1ccccc1",
     "conditions": "Pd/C, H2, 1 atm, rt", "difficulty": "medium"},

    # ============================================================
    # 环化反应 (Cyclization) - Medium
    # ============================================================
    {"type": "cyclization", "reactants_smiles": ["OCCCC(=O)O"], "product_smiles": "O=C1OCCC1",
     "conditions": "H2SO4 cat., toluene, reflux, Dean-Stark", "difficulty": "medium"},
    {"type": "cyclization", "reactants_smiles": ["OCCCCC(=O)O"], "product_smiles": "O=C1OCCCC1",
     "conditions": "H2SO4 cat., toluene, reflux", "difficulty": "medium"},
    {"type": "cyclization", "reactants_smiles": ["NCCCC(=O)O"], "product_smiles": "O=C1NCCCC1",
     "conditions": "heat, 200°C", "difficulty": "medium"},
    {"type": "cyclization", "reactants_smiles": ["OCCCCN"], "product_smiles": "C1CCNCC1",
     "conditions": "H2SO4 cat., 150°C", "difficulty": "medium"},
    {"type": "cyclization", "reactants_smiles": ["OCCCCCO"], "product_smiles": "C1CCOCCC1",
     "conditions": "H2SO4 cat., 140°C", "difficulty": "medium"},

    # ============================================================
    # 酰胺化反应 (Amidation) - Easy
    # ============================================================
    {"type": "amidation", "reactants_smiles": ["CC(=O)O", "CCN"], "product_smiles": "CC(=O)NCC",
     "conditions": "DCC, DCM, 0°C→rt, 12h", "difficulty": "easy"},
    {"type": "amidation", "reactants_smiles": ["CC(=O)Cl", "CCN"], "product_smiles": "CC(=O)NCC",
     "conditions": "Et3N, DCM, 0°C", "difficulty": "easy"},
    {"type": "amidation", "reactants_smiles": ["CC(=O)Cl", "N"], "product_smiles": "CC(=O)N",
     "conditions": "NH3, DCM, 0°C", "difficulty": "easy"},
    {"type": "amidation", "reactants_smiles": ["CC(=O)Cl", "CCCO"], "product_smiles": "CCCOC(=O)C",
     "conditions": "Py, DCM, 0°C", "difficulty": "easy"},

    # ============================================================
    # 酰胺化反应 (Amidation) - Medium
    # ============================================================
    {"type": "amidation", "reactants_smiles": ["OC(=O)c1ccccc1", "CN"], "product_smiles": "CNC(=O)c1ccccc1",
     "conditions": "EDC, HOBt, DMF, rt, 24h", "difficulty": "medium"},
    {"type": "amidation", "reactants_smiles": ["CC(=O)O", "Nc1ccccc1"], "product_smiles": "CC(=O)Nc1ccccc1",
     "conditions": "heat, 180°C", "difficulty": "medium"},
    {"type": "amidation", "reactants_smiles": ["OC(=O)CC(=O)O", "CCN"], "product_smiles": "CCNC(=O)CC(=O)NCC",
     "conditions": "DCC, DCM, 0°C→rt, 24h", "difficulty": "medium"},
    {"type": "amidation", "reactants_smiles": ["OC(=O)c1ccccc1", "CCN"], "product_smiles": "CCNC(=O)c1ccccc1",
     "conditions": "SOCl2, rt, then Et3N, DCM", "difficulty": "medium"},

    # ============================================================
    # Diels-Alder 反应 - Hard
    # ============================================================
    {"type": "Diels-Alder", "reactants_smiles": ["C=CC=C", "C=CC(=O)C"], "product_smiles": "O=C(C)C1CCCC=C1",
     "conditions": "toluene, 110°C, sealed tube", "difficulty": "hard"},
    {"type": "Diels-Alder", "reactants_smiles": ["C=CC=C", "C=CC(=O)OC"], "product_smiles": "O=C(OC)C1CCCC=C1",
     "conditions": "toluene, 100°C", "difficulty": "hard"},
    {"type": "Diels-Alder", "reactants_smiles": ["C=CC=C", "C=CC#N"], "product_smiles": "N#CC1CCCC=C1",
     "conditions": "toluene, 80°C", "difficulty": "hard"},
    {"type": "Diels-Alder", "reactants_smiles": ["C=Cc1ccccc1", "C=CC=C"], "product_smiles": "C1CCc2ccccc2C=C1",
     "conditions": "toluene, 150°C", "difficulty": "hard"},
    {"type": "Diels-Alder", "reactants_smiles": ["C=CC=C", "O=Cc1ccccc1"], "product_smiles": "O=C1CCCC=C1c2ccccc2",
     "conditions": "CHCl3, 60°C", "difficulty": "hard"},

    # ============================================================
    # 水解反应 (Hydrolysis) - Easy
    # ============================================================
    {"type": "hydrolysis", "reactants_smiles": ["CC(=O)OCC", "[OH-]"], "product_smiles": "CC(=O)[O-]",
     "conditions": "NaOH, H2O, reflux", "difficulty": "easy"},
    {"type": "hydrolysis", "reactants_smiles": ["CC(=O)N", "O"], "product_smiles": "CC(=O)O",
     "conditions": "HCl, H2O, reflux, 6h", "difficulty": "easy"},
    {"type": "hydrolysis", "reactants_smiles": ["CC#N"], "product_smiles": "CC(=O)O",
     "conditions": "NaOH, H2O, reflux, 12h", "difficulty": "easy"},
    {"type": "hydrolysis", "reactants_smiles": ["CC(=O)OCC(=O)C"], "product_smiles": "CC(=O)O",
     "conditions": "NaOH, H2O, reflux, 4h", "difficulty": "easy"},

    # ============================================================
    # 水解反应 (Hydrolysis) - Medium
    # ============================================================
    {"type": "hydrolysis", "reactants_smiles": ["CC(=O)NCC"], "product_smiles": "CC(=O)O",
     "conditions": "HCl, H2O, reflux, 6h", "difficulty": "medium"},
    {"type": "hydrolysis", "reactants_smiles": ["CCOC(=O)CC(=O)OCC"], "product_smiles": "OC(=O)CC(=O)O",
     "conditions": "NaOH, H2O, reflux, 8h", "difficulty": "medium"},
    {"type": "hydrolysis", "reactants_smiles": ["CC(=O)Oc1ccccc1"], "product_smiles": "OC(=O)c1ccccc1",
     "conditions": "NaOH, H2O, reflux, 4h", "difficulty": "medium"},

    # ============================================================
    # 卤化反应 (Halogenation) - Easy
    # ============================================================
    {"type": "halogenation", "reactants_smiles": ["c1ccccc1"], "product_smiles": "Brc1ccccc1",
     "conditions": "Br2, FeBr3, 25°C", "difficulty": "easy"},
    {"type": "halogenation", "reactants_smiles": ["c1ccccc1"], "product_smiles": "O=[N+]([O-])c1ccccc1",
     "conditions": "HNO3/H2SO4, 50°C", "difficulty": "easy"},
    {"type": "halogenation", "reactants_smiles": ["CC"], "product_smiles": "CCBr",
     "conditions": "Br2, hv, 25°C", "difficulty": "easy"},
    {"type": "halogenation", "reactants_smiles": ["c1ccccc1"], "product_smiles": "Clc1ccccc1",
     "conditions": "Cl2, FeCl3, 25°C", "difficulty": "easy"},

    # ============================================================
    # 卤化反应 (Halogenation) - Medium
    # ============================================================
    {"type": "halogenation", "reactants_smiles": ["c1ccccc1C"], "product_smiles": "O=[N+]([O-])c1ccccc1C",
     "conditions": "HNO3/H2SO4, 30°C", "difficulty": "medium"},
    {"type": "halogenation", "reactants_smiles": ["c1ccccc1"], "product_smiles": "O=[N+]([O-])c1ccccc1[N+](=O)[O-]",
     "conditions": "HNO3/H2SO4, 100°C", "difficulty": "medium"},
    {"type": "halogenation", "reactants_smiles": ["CC(=O)c1ccccc1"], "product_smiles": "CC(=O)c1cccc(Br)c1",
     "conditions": "Br2, AcOH, 25°C", "difficulty": "medium"},

    # ============================================================
    # Friedel-Crafts 反应 - Easy
    # ============================================================
    {"type": "Friedel-Crafts", "reactants_smiles": ["c1ccccc1", "CCCl"], "product_smiles": "CCc1ccccc1",
     "conditions": "AlCl3, 25°C", "difficulty": "easy"},
    {"type": "Friedel-Crafts", "reactants_smiles": ["c1ccccc1", "CC(=O)Cl"], "product_smiles": "CC(=O)c1ccccc1",
     "conditions": "AlCl3, DCM, 0°C→rt", "difficulty": "easy"},
    {"type": "Friedel-Crafts", "reactants_smiles": ["c1ccccc1", "CCCCl"], "product_smiles": "CCCc1ccccc1",
     "conditions": "AlCl3, 25°C", "difficulty": "easy"},

    # ============================================================
    # Friedel-Crafts 反应 - Medium
    # ============================================================
    {"type": "Friedel-Crafts", "reactants_smiles": ["c1ccccc1", "CC(=O)O"], "product_smiles": "CC(=O)c1ccccc1",
     "conditions": "AlCl3, PCl3, 160°C", "difficulty": "medium"},
    {"type": "Friedel-Crafts", "reactants_smiles": ["c1ccccc1", "CC(=O)OC"], "product_smiles": "CC(=O)c1ccccc1",
     "conditions": "AlCl3, 80°C", "difficulty": "medium"},
    {"type": "Friedel-Crafts", "reactants_smiles": ["CC(=O)c1ccccc1", "CCCl"], "product_smiles": "CC(=O)c1cccc(CC)c1",
     "conditions": "AlCl3, 25°C", "difficulty": "medium"},

    # ============================================================
    # Grignard 反应 - Easy
    # ============================================================
    {"type": "Grignard", "reactants_smiles": ["CMgBr", "C=O"], "product_smiles": "CO",
     "conditions": "Et2O, 0°C, then H3O+", "difficulty": "easy"},
    {"type": "Grignard", "reactants_smiles": ["CCMgBr", "CC=O"], "product_smiles": "CCC(C)O",
     "conditions": "THF, 0°C→rt, then H3O+", "difficulty": "easy"},
    {"type": "Grignard", "reactants_smiles": ["CMgBr", "CC=O"], "product_smiles": "CC(C)O",
     "conditions": "Et2O, 0°C, then H3O+", "difficulty": "easy"},

    # ============================================================
    # Grignard 反应 - Medium
    # ============================================================
    {"type": "Grignard", "reactants_smiles": ["CCMgBr", "CC(=O)C"], "product_smiles": "CCC(C)(C)O",
     "conditions": "Et2O, 0°C, then H3O+", "difficulty": "medium"},
    {"type": "Grignard", "reactants_smiles": ["c1ccccc1MgBr", "CC=O"], "product_smiles": "CC(O)c1ccccc1",
     "conditions": "THF, 0°C, then H3O+", "difficulty": "medium"},
    {"type": "Grignard", "reactants_smiles": ["CCMgBr", "C=O"], "product_smiles": "CC(C)O",
     "conditions": "Et2O, 0°C, then H3O+", "difficulty": "medium"},
    {"type": "Grignard", "reactants_smiles": ["c1ccccc1MgBr", "CC(=O)C"], "product_smiles": "CC(C)(O)c1ccccc1",
     "conditions": "THF, 0°C, then H3O+", "difficulty": "medium"},

    # ============================================================
    # Grignard 反应 - Hard
    # ============================================================
    {"type": "Grignard", "reactants_smiles": ["CCMgBr", "OC(=O)c1ccccc1"], "product_smiles": "CC(C)(O)c1ccccc1",
     "conditions": "THF, 0°C, then H3O+", "difficulty": "hard"},
    {"type": "Grignard", "reactants_smiles": ["CCMgBr", "CC(=O)OCC"], "product_smiles": "CCC(C)(C)O",
     "conditions": "Et2O, 0°C, then H3O+", "difficulty": "hard"},
    {"type": "Grignard", "reactants_smiles": ["c1ccccc1MgBr", "CC(=O)O"], "product_smiles": "CC(C)(O)c1ccccc1",
     "conditions": "THF, -78°C, then H3O+", "difficulty": "hard"},
]


def format_reaction_input(reaction: dict) -> str:
    """格式化反应输入文本"""
    reactants_str = " + ".join(reaction["reactants_smiles"])
    return (
        f"Predict the major organic product of the following reaction.\n"
        f"Reaction type: {reaction['type']}\n"
        f"Reactants: {reactants_str}\n"
        f"Conditions: {reaction['conditions']}\n"
        f"Difficulty: {reaction['difficulty']}\n"
        f"Provide the product SMILES."
    )


def prepare_consistent_100_eval_dataset():
    """
    准备100样本评估集 - 确保训练集和评估集反应类型一致
    """
    all_reactions = REAL_REACTIONS.copy()
    print(f"总反应数: {len(all_reactions)}")

    # 统计反应类型
    reaction_types = set(r["type"] for r in all_reactions)
    print(f"反应类型数: {len(reaction_types)}")
    print(f"反应类型: {sorted(reaction_types)}")

    # 按难度分组
    easy = [r for r in all_reactions if r["difficulty"] == "easy"]
    medium = [r for r in all_reactions if r["difficulty"] == "medium"]
    hard = [r for r in all_reactions if r["difficulty"] == "hard"]

    print(f"\n按难度统计:")
    print(f"  Easy: {len(easy)}")
    print(f"  Medium: {len(medium)}")
    print(f"  Hard: {len(hard)}")

    # 目标评估集: 40 easy + 40 medium + 20 hard = 100
    # 确保每种反应类型在训练集和评估集中都有
    target_eval = {"easy": min(40, len(easy)), "medium": min(40, len(medium)), "hard": min(20, len(hard))}

    random.shuffle(easy)
    random.shuffle(medium)
    random.shuffle(hard)

    # 确保每种反应类型至少在训练集中出现一次
    def split_by_type(reactions, eval_count):
        """按类型分组，确保每种类型都有训练和评估样本"""
        type_groups = {}
        for r in reactions:
            t = r["type"]
            if t not in type_groups:
                type_groups[t] = []
            type_groups[t].append(r)

        eval_samples = []
        train_samples = []

        # 每种类型至少分配1个到评估集
        for t, samples in type_groups.items():
            random.shuffle(samples)
            # 分配至少1个到评估集，但不超过总数的50%
            eval_for_type = max(1, min(len(samples) // 2, eval_count // len(type_groups)))
            eval_samples.extend(samples[:eval_for_type])
            train_samples.extend(samples[eval_for_type:])

        # 如果评估集还不够，从训练集中补充
        while len(eval_samples) < eval_count and train_samples:
            eval_samples.append(train_samples.pop())

        return train_samples, eval_samples

    train_easy, eval_easy = split_by_type(easy, target_eval["easy"])
    train_medium, eval_medium = split_by_type(medium, target_eval["medium"])
    train_hard, eval_hard = split_by_type(hard, target_eval["hard"])

    train_data_raw = train_easy + train_medium + train_hard
    eval_data_raw = eval_easy + eval_medium + eval_hard

    random.shuffle(train_data_raw)
    random.shuffle(eval_data_raw)

    # 转换为训练格式
    def to_training_format(reactions, start_idx=0):
        samples = []
        for i, r in enumerate(reactions):
            samples.append({
                "id": f"rxn_{start_idx + i:04d}",
                "reaction_type": r["type"],
                "reactants_smiles": r["reactants_smiles"],
                "product_smiles": r["product_smiles"],
                "conditions": r["conditions"],
                "difficulty": r["difficulty"],
                "input_text": format_reaction_input(r),
                "target_text": r["product_smiles"],
            })
        return samples

    train_samples = to_training_format(train_data_raw, 0)
    eval_samples = to_training_format(eval_data_raw, len(train_samples))

    # 保存
    train_path = OUTPUT_DIR / "train_reactions_100_consistent.json"
    eval_path = OUTPUT_DIR / "eval_reactions_100_consistent.json"
    metadata_path = OUTPUT_DIR / "dataset_metadata_100_consistent.json"

    with open(train_path, "w") as f:
        json.dump(train_samples, f, indent=2)
    with open(eval_path, "w") as f:
        json.dump(eval_samples, f, indent=2)

    # 验证反应类型一致性
    train_types = set(r["reaction_type"] for r in train_samples)
    eval_types = set(r["reaction_type"] for r in eval_samples)
    eval_only_types = eval_types - train_types

    metadata = {
        "version": "100_eval_consistent",
        "total_examples": len(all_reactions),
        "train_size": len(train_samples),
        "eval_size": len(eval_samples),
        "difficulty_distribution": {
            "train": {"easy": len(train_easy), "medium": len(train_medium), "hard": len(train_hard)},
            "eval": {"easy": len(eval_easy), "medium": len(eval_medium), "hard": len(eval_hard)},
        },
        "reaction_types": sorted(list(set(r["type"] for r in all_reactions))),
        "num_reaction_types": len(set(r["type"] for r in all_reactions)),
        "evaluation": "exact_SMILES_match",
        "eval_target_size": 100,
        "seed": 42,
        "type_consistency_check": {
            "train_types": sorted(list(train_types)),
            "eval_types": sorted(list(eval_types)),
            "eval_only_types": sorted(list(eval_only_types)),
            "consistent": len(eval_only_types) == 0
        }
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'='*60}")
    print(f"100样本评估集准备完成!（类型一致性保证）")
    print(f"{'='*60}")
    print(f"训练集: {len(train_samples)} 样本")
    print(f"  - Easy: {len(train_easy)}, Medium: {len(train_medium)}, Hard: {len(train_hard)}")
    print(f"评估集: {len(eval_samples)} 样本")
    print(f"  - Easy: {len(eval_easy)}, Medium: {len(eval_medium)}, Hard: {len(eval_hard)}")
    print(f"反应类型数: {metadata['num_reaction_types']}")
    print(f"类型一致性: {'✓ 通过' if len(eval_only_types) == 0 else '✗ 失败'}")
    if eval_only_types:
        print(f"  ⚠ 只在评估集中的类型: {eval_only_types}")

    return train_samples, eval_samples


if __name__ == "__main__":
    print("=" * 60)
    print("RSI MVE: 100样本评估集准备（类型一致性保证）")
    print("=" * 60)
    prepare_consistent_100_eval_dataset()
