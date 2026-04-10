"""
Procedural scenario generator for the customs-clearance-env.

Given a task_id and a numeric seed, produces a unique shipment scenario
with deterministic ground-truth answers. Complements the 24 hardcoded
scenarios in documents.py with effectively unlimited seed-based variety.
"""
from __future__ import annotations

import math
import random
from typing import Any

# ── Constants ─────────────────────────────────────────────────────────────────

FX_RATE_USD_INR = 83.0
FREIGHT_PCT = 0.04
INSURANCE_PCT = 0.0125

# ── Commodity table ───────────────────────────────────────────────────────────
# (description, hs_code, duty_rate, typical_unit_value_usd, category)

COMMODITIES: list[tuple[str, str, float, float, str]] = [
    ("Wireless Bluetooth headphones", "8518.30.00", 0.20, 24.0, "electronics"),
    ("USB Type-C charging cables, copper conductors", "8544.42.90", 0.15, 0.85, "electronics"),
    ("Smartphone tempered glass screen protectors", "7007.19.00", 0.10, 0.12, "electronics"),
    ("LCD touch assemblies for smartphones", "8517.70.90", 0.10, 16.0, "electronics"),
    ("LED lamps for general lighting, household type", "9405.40.90", 0.15, 3.0, "electronics"),
    ("Laptop cooling pads with USB fans", "8414.59.90", 0.15, 3.0, "electronics"),
    ("Ethernet patch cords Cat6", "8544.42.90", 0.15, 1.5, "electronics"),
    ("Portable Bluetooth speakers", "8518.22.00", 0.20, 30.0, "electronics"),
    ("CCTV surveillance cameras, IP type", "8525.89.00", 0.15, 22.0, "electronics"),
    ("Solar panel modules, polycrystalline", "8541.40.00", 0.00, 55.0, "electronics"),
    ("Men's knitted cotton T-shirts, 100% cotton", "6109.10.00", 0.35, 4.0, "textiles"),
    ("Women's woven polyester blouses", "6206.40.00", 0.35, 6.5, "textiles"),
    ("Cotton bed linen sets, printed", "6302.21.00", 0.25, 8.0, "textiles"),
    ("Synthetic yarn for knitting, polyester", "5402.33.00", 0.15, 2.2, "textiles"),
    ("Organic chemical intermediate for industrial synthesis", "2921.41.00", 0.15, 22.5, "chemicals"),
    ("Anthranilic acid derivatives (controlled precursor)", "2922.43.00", 0.15, 35.0, "chemicals"),
    ("PVC resin suspension grade", "3904.10.00", 0.10, 1.1, "chemicals"),
    ("Titanium dioxide pigment", "2823.00.00", 0.10, 2.8, "chemicals"),
    ("Industrial machinery parts: indexing tables and rotary feeders", "8479.89.90", 0.15, 3700.0, "machinery"),
    ("CNC lathe spindle assemblies", "8466.93.00", 0.10, 4200.0, "machinery"),
    ("Hydraulic press cylinders, rated >100 tons", "8412.21.00", 0.10, 2800.0, "machinery"),
    ("Steel wood screws, countersunk head", "7318.12.00", 0.10, 0.044, "hardware"),
    ("Reinforced rubber hoses with fittings for hydraulic systems", "4009.21.00", 0.10, 30.0, "hardware"),
    ("Stainless steel kitchen sinks", "7324.10.00", 0.15, 80.0, "hardware"),
    ("Cast iron cookware sets", "7323.91.00", 0.10, 30.0, "hardware"),
    ("Basmati rice, milled, long grain", "1006.30.00", 0.00, 1.2, "food"),
    ("Frozen shrimp, peeled and deveined", "0306.17.00", 0.30, 12.0, "food"),
    ("Olive oil, extra virgin, in bulk containers", "1509.10.00", 0.45, 5.5, "food"),
    ("Ibuprofen bulk powder API", "2942.00.00", 0.10, 18.0, "pharma"),
    ("Empty hard gelatin capsules, size 0", "3926.90.99", 0.10, 0.008, "pharma"),
]

# ── Party pools ───────────────────────────────────────────────────────────────

FOREIGN_SHIPPERS = [
    ("Acme Electronics Co Ltd", "Shanghai"),
    ("Shenzhen Cable Manufacturing", "Shenzhen"),
    ("Guangzhou Apparel Ltd", "Guangzhou"),
    ("Foxconn Precision Components", "Taipei"),
    ("Jiangsu Heavy Parts Co", "Nanjing"),
    ("Ningbo Lighting Export", "Ningbo"),
    ("Tianjin Fasteners Ltd", "Tianjin"),
    ("Qingdao Rubber Products", "Qingdao"),
    ("Hanoi Textile Corp", "Hanoi"),
    ("Bangkok Chemicals Co", "Bangkok"),
    ("Osaka Machinery Inc", "Osaka"),
    ("Hamburg Chemie GmbH", "Hamburg"),
    ("Istanbul Trading SA", "Istanbul"),
    ("Dubai Global Logistics FZE", "Dubai"),
    ("Seoul Semiconductor Co", "Seoul"),
]

INDIAN_CONSIGNEES = [
    ("Tech Imports Pvt Ltd", "Chennai"),
    ("Spark Retail India", "Mumbai"),
    ("Urban Style India LLP", "Bengaluru"),
    ("Mobile Assembly India Pvt Ltd", "Noida"),
    ("Precision Tools India", "Pune"),
    ("GreenLite Distributors", "Kochi"),
    ("BuildWell Hardware", "Ahmedabad"),
    ("Hydraulics India Pvt Ltd", "Chennai"),
    ("Nova Electronics Pvt Ltd", "New Delhi"),
    ("MedPharma Supplies Ltd", "Hyderabad"),
]

LOAD_PORTS = [
    ("CNSHA", "China"), ("CNSZX", "China"), ("CNGGZ", "China"),
    ("CNTXG", "China"), ("CNNKG", "China"), ("CNNBO", "China"),
    ("CNQDG", "China"), ("CNXMN", "China"), ("HKHKG", "Hong Kong"),
    ("KRPUS", "South Korea"), ("JPYOK", "Japan"), ("DEHAM", "Germany"),
    ("VNHPH", "Vietnam"), ("THBKK", "Thailand"), ("TRIST", "Turkey"),
    ("AEJER", "UAE"), ("USNYC", "USA"),
]

DISCHARGE_PORTS = [
    "INMAA", "INNSA", "INBLR", "INDRI", "INPNQ",
    "INCOK", "INAMD", "INHZA",
]

COUNTRY_FOR_PORT_PREFIX = {
    "CN": "China", "HK": "Hong Kong", "KR": "South Korea",
    "JP": "Japan", "DE": "Germany", "VN": "Vietnam",
    "TH": "Thailand", "TR": "Turkey", "AE": "UAE", "US": "USA",
    "BD": "Bangladesh", "IN": "India",
}

# ── Error recipes for task2 / task3 ──────────────────────────────────────────
# Each recipe: (flag_id, mutator function, compatibility group)
# Mutators operate on (invoice, packing_list, bl, rng) -> None (in-place)

def _mutate_qty_mismatch(inv: dict, pl: dict, bl: dict, rng: random.Random) -> None:
    pl["quantity"] = int(inv["quantity"] * rng.uniform(0.82, 0.95))

def _mutate_missing_origin(inv: dict, pl: dict, bl: dict, rng: random.Random) -> None:
    inv.pop("country_of_origin", None)

def _mutate_weight_mismatch(inv: dict, pl: dict, bl: dict, rng: random.Random) -> None:
    base_weight = inv["quantity"] * rng.uniform(0.8, 2.5)
    pl["gross_weight_kg"] = round(base_weight, 1)
    bl["gross_weight_kg"] = round(base_weight * rng.uniform(0.78, 0.92), 1)

def _mutate_invoice_number_mismatch(inv: dict, pl: dict, bl: dict, rng: random.Random) -> None:
    bl["invoice_number"] = inv["invoice_number"] + "-" + str(rng.randint(1, 9))

def _mutate_missing_bl_invoice(inv: dict, pl: dict, bl: dict, rng: random.Random) -> None:
    bl.pop("invoice_number", None)

def _mutate_goods_desc_mismatch(inv: dict, pl: dict, bl: dict, rng: random.Random) -> None:
    substitutes = ["Ceramic bathroom fixtures", "Plastic household containers",
                    "Metal office furniture parts", "Synthetic textile remnants"]
    pl["goods_description"] = rng.choice(substitutes)

def _mutate_consignee_typo(inv: dict, pl: dict, bl: dict, rng: random.Random) -> None:
    if "consignee" in bl and len(bl["consignee"]) > 6:
        name = bl["consignee"]
        pos = rng.randint(3, len(name) - 3)
        bl["consignee"] = name[:pos] + name[pos + 1:]

def _mutate_missing_notify(inv: dict, pl: dict, bl: dict, rng: random.Random) -> None:
    bl["notify_party"] = ""

def _mutate_undervaluation(inv: dict, pl: dict, bl: dict, rng: random.Random) -> None:
    inv["declared_value_usd"] = round(inv["declared_value_usd"] * rng.uniform(0.08, 0.25), 2)

ERROR_RECIPES: list[tuple[str, Any, str]] = [
    ("quantity_mismatch", _mutate_qty_mismatch, "qty"),
    ("missing_country_of_origin", _mutate_missing_origin, "origin"),
    ("weight_mismatch_packing_vs_bl", _mutate_weight_mismatch, "weight"),
    ("invoice_number_mismatch_bl_vs_invoice", _mutate_invoice_number_mismatch, "inv_num"),
    ("missing_invoice_number_on_bl", _mutate_missing_bl_invoice, "inv_num"),
    ("goods_description_mismatch_invoice_vs_packing_list", _mutate_goods_desc_mismatch, "goods"),
    ("consignee_name_mismatch", _mutate_consignee_typo, "consignee"),
    ("missing_notify_party", _mutate_missing_notify, "notify"),
    ("suspected_undervaluation", _mutate_undervaluation, "value"),
]

TASK3_EXTRA_FLAGS = [
    "vague_goods_description",
    "origin_loading_mismatch",
    "high_value_shipment",
    "mixed_consignment_requires_classification",
    "textile_declaration_review",
    "multilingual_document_review",
    "dual_use_or_controlled_chemical_risk",
    "ambiguous_end_use",
]


def _derive_recommendation(flags: list[str]) -> str:
    flag_set = set(flags)
    if flag_set & {"dual_use_or_controlled_chemical_risk", "ambiguous_end_use"}:
        return "refer_to_customs"
    if not flags:
        return "clear"
    if flag_set & {"suspected_undervaluation"} and len(flags) <= 2:
        return "query_shipper"
    return "hold"


def _port_country(port_code: str) -> str:
    return COUNTRY_FOR_PORT_PREFIX.get(port_code[:2], "Unknown")


# ── Generator functions ───────────────────────────────────────────────────────

def generate_task1(seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    commodity = rng.choice(COMMODITIES)
    desc, hs, _duty, unit_val, _cat = commodity
    shipper_name, shipper_city = rng.choice(FOREIGN_SHIPPERS)
    consignee_name, consignee_city = rng.choice(INDIAN_CONSIGNEES)
    load_port, _country = rng.choice(LOAD_PORTS)
    discharge_port = rng.choice(DISCHARGE_PORTS)
    qty = rng.choice([100, 200, 500, 1000, 2000, 5000, 10000, 20000])
    declared_value = round(qty * unit_val * rng.uniform(0.85, 1.15), 2)
    inv_num = f"INV-G{seed:05d}"

    content = {
        "shipper": f"{shipper_name}, {shipper_city}",
        "consignee": f"{consignee_name}, {consignee_city}",
        "invoice_number": inv_num,
        "goods_description": desc,
        "quantity": qty,
        "declared_value": declared_value,
        "currency": "USD",
        "country_of_origin": _port_country(load_port),
        "port_of_loading": load_port,
        "port_of_discharge": discharge_port,
    }
    return {
        "id": f"gen_t1_{seed}",
        "document_type": "invoice",
        "document_content": content,
        "task_instruction": "Assign the correct 8-digit HS code for the goods on this commercial invoice.",
        "correct_answer": {"hs_code": hs},
    }


def generate_task2(seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    commodity = rng.choice(COMMODITIES)
    desc, _hs, _duty, unit_val, _cat = commodity
    shipper_name, shipper_city = rng.choice(FOREIGN_SHIPPERS)
    consignee_name, consignee_city = rng.choice(INDIAN_CONSIGNEES)
    load_port, _country = rng.choice(LOAD_PORTS)
    discharge_port = rng.choice(DISCHARGE_PORTS)
    qty = rng.choice([200, 500, 800, 1200, 2000, 5000, 8000, 50000])
    declared_value_usd = round(qty * unit_val * rng.uniform(0.85, 1.15), 2)
    inv_num = f"INV-G{seed:05d}"
    bl_num = f"MAEU{rng.randint(100000000, 999999999)}"

    invoice = {
        "type": "invoice",
        "invoice_number": inv_num,
        "goods_description": desc,
        "quantity": qty,
        "declared_value_usd": declared_value_usd,
        "country_of_origin": _port_country(load_port),
    }
    packing_list: dict[str, Any] = {
        "type": "packing_list",
        "quantity": qty,
    }
    bill_of_lading: dict[str, Any] = {
        "type": "bill_of_lading",
        "bl_number": bl_num,
        "invoice_number": inv_num,
        "consignee": f"{consignee_name}, {consignee_city}",
        "port_of_loading": load_port,
        "port_of_discharge": discharge_port,
    }

    num_errors = rng.choices([1, 2, 3], weights=[0.35, 0.45, 0.20])[0]
    available = list(ERROR_RECIPES)
    rng.shuffle(available)
    chosen_groups: set[str] = set()
    chosen_flags: list[str] = []
    for flag_id, mutator, group in available:
        if len(chosen_flags) >= num_errors:
            break
        if group in chosen_groups:
            continue
        mutator(invoice, packing_list, bill_of_lading, rng)
        chosen_flags.append(flag_id)
        chosen_groups.add(group)

    recommendation = _derive_recommendation(chosen_flags)

    return {
        "id": f"gen_t2_{seed}",
        "document_type": "shipment_file",
        "document_content": {"documents": [invoice, packing_list, bill_of_lading]},
        "task_instruction": "Review Invoice, Packing List, and Bill of Lading. Flag every compliance issue you find.",
        "correct_answer": {"flags": chosen_flags, "recommendation": recommendation},
    }


def generate_task3(seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    commodity = rng.choice(COMMODITIES)
    desc, hs, duty_rate, unit_val, cat = commodity
    shipper_name, shipper_city = rng.choice(FOREIGN_SHIPPERS)
    consignee_name, consignee_city = rng.choice(INDIAN_CONSIGNEES)
    load_port, load_country = rng.choice(LOAD_PORTS)
    discharge_port = rng.choice(DISCHARGE_PORTS)
    qty = rng.choice([50, 200, 500, 1000, 2000, 4000, 10000])
    declared_value_usd = round(qty * unit_val * rng.uniform(0.85, 1.15), 2)
    inv_num = f"INV-G{seed:05d}"
    bl_num = f"HLCU{rng.randint(100000000, 999999999)}"

    declared_origin = load_country
    if rng.random() < 0.25:
        declared_origin = rng.choice(["Bangladesh", "Vietnam", "Thailand", "Indonesia"])

    invoice: dict[str, Any] = {
        "type": "invoice",
        "invoice_number": inv_num,
        "goods_description": desc,
        "quantity": qty,
        "declared_value_usd": declared_value_usd,
        "country_of_origin": declared_origin,
    }
    packing_list: dict[str, Any] = {"type": "packing_list", "quantity": qty}
    bill_of_lading: dict[str, Any] = {
        "type": "bill_of_lading",
        "bl_number": bl_num,
        "invoice_number": inv_num,
        "port_of_loading": load_port,
        "port_of_discharge": discharge_port,
    }

    documents: list[dict[str, Any]] = [invoice, packing_list, bill_of_lading]
    flags: list[str] = []

    num_doc_errors = rng.choices([0, 1, 2], weights=[0.25, 0.50, 0.25])[0]
    available = list(ERROR_RECIPES)
    rng.shuffle(available)
    used_groups: set[str] = set()
    for flag_id, mutator, group in available:
        if len(flags) >= num_doc_errors:
            break
        if group in used_groups:
            continue
        mutator(invoice, packing_list, bill_of_lading, rng)
        flags.append(flag_id)
        used_groups.add(group)

    if declared_origin != _port_country(load_port):
        flags.append("origin_loading_mismatch")

    if cat == "chemicals":
        if rng.random() < 0.6:
            flags.append("dual_use_or_controlled_chemical_risk")
            documents.append({
                "type": "msds",
                "substance": desc.split(",")[0],
                "hazard_class": rng.choice(["irritant", "corrosive", "toxic", "flammable"]),
            })

    if declared_value_usd > 50_000:
        flags.append("high_value_shipment")

    if len(desc.split()) <= 3 or "accessories" in desc.lower() or "parts" in desc.lower():
        if "vague_goods_description" not in flags:
            flags.append("vague_goods_description")

    if cat == "textiles" and declared_origin in ("Bangladesh", "Vietnam", "Indonesia"):
        flags.append("textile_declaration_review")

    flags = list(dict.fromkeys(flags))
    recommendation = _derive_recommendation(flags)

    assessable_inr = round(declared_value_usd * FX_RATE_USD_INR * (1 + FREIGHT_PCT + INSURANCE_PCT), 2)
    duty_inr = round(assessable_inr * duty_rate, 2)

    follow_up_reveals = {
        "detailed_goods_description": f"Detailed: {desc} — {qty} units, {cat} category, HS chapter {hs[:4]}",
        "certificate_of_origin": f"CoO issued by chamber of commerce in {declared_origin}",
        "exchange_rate": f"Applied FX rate: 1 USD = {FX_RATE_USD_INR} INR",
        "duty_rate_schedule": f"Applicable BCD rate for HS {hs}: {duty_rate*100:.0f}%",
    }

    return {
        "id": f"gen_t3_{seed}",
        "document_type": "shipment_file",
        "document_content": {"documents": documents},
        "task_instruction": (
            "Classify goods, flag anomalies, recommend clearance, "
            "and estimate assessable value and duty (INR)."
        ),
        "correct_answer": {
            "hs_code": hs,
            "flags": flags,
            "recommendation": recommendation,
            "assessable_value_inr": assessable_inr,
            "duty_amount_inr": duty_inr,
        },
        "follow_up_reveals": follow_up_reveals,
        "max_steps": 3,
    }


_GENERATORS = {
    "task1": generate_task1,
    "task2": generate_task2,
    "task3": generate_task3,
}


def generate_scenario(task_id: str, seed: int) -> dict[str, Any]:
    gen = _GENERATORS.get(task_id)
    if gen is None:
        raise ValueError(f"Unknown task_id for generation: {task_id}")
    return gen(seed)
