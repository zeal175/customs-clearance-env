"""
Synthetic sea-freight shipment documents for CHA (Custom House Agent) training.
Ground-truth answers are aligned with Indian HS nomenclature-style codes (8-digit dots).
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict


class GroundTruthTask1(TypedDict):
    hs_code: str


class GroundTruthTask2(TypedDict):
    flags: list[str]
    recommendation: Literal["clear", "hold", "query_shipper", "refer_to_customs"]


class GroundTruthTask3(TypedDict):
    hs_code: str
    flags: list[str]
    recommendation: Literal["clear", "hold", "query_shipper", "refer_to_customs"]
    assessable_value_inr: float
    duty_amount_inr: float


def _t1(
    doc_id: str,
    content: dict[str, Any],
    hs: str,
    instruction: str | None = None,
) -> dict[str, Any]:
    return {
        "id": doc_id,
        "document_type": "invoice",
        "document_content": content,
        "task_instruction": instruction
        or "Assign the correct 8-digit HS code for the goods on this commercial invoice.",
        "correct_answer": GroundTruthTask1(hs_code=hs),
    }


def _t2(
    doc_id: str,
    documents: list[dict[str, Any]],
    flags: list[str],
    recommendation: str,
    instruction: str | None = None,
) -> dict[str, Any]:
    return {
        "id": doc_id,
        "document_type": "shipment_file",
        "document_content": {"documents": documents},
        "task_instruction": instruction
        or "Review Invoice, Packing List, and Bill of Lading. Flag every compliance issue you find.",
        "correct_answer": GroundTruthTask2(
            flags=flags,
            recommendation=recommendation,  # type: ignore[arg-type]
        ),
    }


def _t3(
    doc_id: str,
    content: dict[str, Any],
    hs: str,
    flags: list[str],
    recommendation: str,
    assessable_inr: float,
    duty_inr: float,
    instruction: str | None = None,
) -> dict[str, Any]:
    return {
        "id": doc_id,
        "document_type": "shipment_file",
        "document_content": content,
        "task_instruction": instruction
        or "Classify goods, flag anomalies, recommend clearance, and estimate assessable value and duty (INR).",
        "correct_answer": GroundTruthTask3(
            hs_code=hs,
            flags=flags,
            recommendation=recommendation,  # type: ignore[arg-type]
            assessable_value_inr=assessable_inr,
            duty_amount_inr=duty_inr,
        ),
    }


# --- Task 1: single clean invoices (easy) ---

TASK1_DOCUMENTS: list[dict[str, Any]] = [
    _t1(
        "t1_headphones",
        {
            "shipper": "Acme Electronics, Shanghai",
            "consignee": "Tech Imports Pvt Ltd, Chennai",
            "invoice_number": "INV-ACM-24001",
            "goods_description": "Wireless Bluetooth Headphones",
            "quantity": 500,
            "declared_value": 12000,
            "currency": "USD",
            "country_of_origin": "China",
            "port_of_loading": "CNSHA",
            "port_of_discharge": "INMAA",
        },
        "8518.30.00",
    ),
    _t1(
        "t1_usb_cables",
        {
            "shipper": "Shenzhen Cable Co Ltd",
            "consignee": "Spark Retail India, Mumbai",
            "invoice_number": "SC-88912",
            "goods_description": "USB Type-C charging cables, copper conductors, molded plugs",
            "quantity": 10000,
            "declared_value": 8500,
            "currency": "USD",
            "country_of_origin": "China",
            "port_of_loading": "CNSZX",
            "port_of_discharge": "INNSA",
        },
        "8544.42.90",
    ),
    _t1(
        "t1_tshirts",
        {
            "shipper": "Guangzhou Apparel Ltd",
            "consignee": "Urban Style India LLP, Bengaluru",
            "invoice_number": "GA-44102",
            "goods_description": "Men's knitted cotton T-shirts, 100% cotton",
            "quantity": 2400,
            "declared_value": 9600,
            "currency": "USD",
            "country_of_origin": "China",
            "port_of_loading": "CNGGZ",
            "port_of_discharge": "INBLR",
        },
        "6109.10.00",
    ),
    _t1(
        "t1_phone_parts",
        {
            "shipper": "Foxconn Precision Components",
            "consignee": "Mobile Assembly India Pvt Ltd, Noida",
            "invoice_number": "FPC-77221",
            "goods_description": "LCD touch assemblies for smartphones (parts)",
            "quantity": 8000,
            "declared_value": 128000,
            "currency": "USD",
            "country_of_origin": "China",
            "port_of_loading": "CNTXG",
            "port_of_discharge": "INDRI",
        },
        "8517.70.90",
    ),
    _t1(
        "t1_machinery_parts",
        {
            "shipper": "Jiangsu Heavy Parts Co",
            "consignee": "Precision Tools India, Pune",
            "invoice_number": "JHP-33091",
            "goods_description": "Industrial machinery parts: indexing tables and rotary feeders (not elsewhere specified)",
            "quantity": 12,
            "declared_value": 44000,
            "currency": "USD",
            "country_of_origin": "China",
            "port_of_loading": "CNNKG",
            "port_of_discharge": "INPNQ",
        },
        "8479.89.90",
    ),
    _t1(
        "t1_led_lamps",
        {
            "shipper": "Ningbo Lighting Export Co",
            "consignee": "GreenLite Distributors, Kochi",
            "invoice_number": "NLE-22018",
            "goods_description": "LED lamps for general lighting, household type",
            "quantity": 6000,
            "declared_value": 18000,
            "currency": "USD",
            "country_of_origin": "China",
            "port_of_loading": "CNNBO",
            "port_of_discharge": "INCOK",
        },
        "9405.40.90",
    ),
    _t1(
        "t1_steel_screws",
        {
            "shipper": "Tianjin Fasteners Ltd",
            "consignee": "BuildWell Hardware, Ahmedabad",
            "invoice_number": "TF-99102",
            "goods_description": "Steel wood screws, countersunk head",
            "quantity": 500000,
            "declared_value": 22000,
            "currency": "USD",
            "country_of_origin": "China",
            "port_of_loading": "CNTXG",
            "port_of_discharge": "INAMD",
        },
        "7318.12.00",
    ),
    _t1(
        "t1_rubber_hoses",
        {
            "shipper": "Qingdao Rubber Products",
            "consignee": "Hydraulics India Pvt Ltd, Chennai",
            "invoice_number": "QR-55100",
            "goods_description": "Reinforced rubber hoses with fittings for hydraulic systems",
            "quantity": 900,
            "declared_value": 27000,
            "currency": "USD",
            "country_of_origin": "China",
            "port_of_loading": "CNQDG",
            "port_of_discharge": "INMAA",
        },
        "4009.21.00",
    ),
]

# --- Task 2: three documents with planted errors ---

TASK2_DOCUMENTS: list[dict[str, Any]] = [
    _t2(
        "t2_qty_mismatch_missing_bl_field",
        [
            {
                "type": "invoice",
                "invoice_number": "INV-HK-771",
                "goods_description": "Bluetooth portable speakers",
                "quantity": 1200,
                "declared_value_usd": 36000,
                "country_of_origin": "China",
            },
            {
                "type": "packing_list",
                "quantity": 1100,
                "cartons": 55,
                "net_weight_kg": 1320,
            },
            {
                "type": "bill_of_lading",
                "bl_number": "MAEU987654321",
                "shipper": "HK Audio Ltd",
                "consignee": "Sound House India, Chennai",
                "port_of_loading": "HKHKG",
                "port_of_discharge": "INMAA",
                # deliberate: no invoice_number on B/L
            },
        ],
        ["quantity_mismatch", "missing_invoice_number_on_bl"],
        "hold",
    ),
    _t2(
        "t2_undervalue_origin",
        [
            {
                "type": "invoice",
                "invoice_number": "INV-SZ-2201",
                "goods_description": "Smartphone tempered glass screen protectors",
                "quantity": 50000,
                "declared_value_usd": 2500,
                "country_of_origin": "China",
            },
            {
                "type": "packing_list",
                "quantity": 50000,
                "cartons": 200,
            },
            {
                "type": "bill_of_lading",
                "bl_number": "CMDU1234567",
                "invoice_number": "INV-SZ-2201",
                "port_of_loading": "CNSZX",
                "port_of_discharge": "INNSA",
            },
        ],
        ["suspected_undervaluation"],
        "query_shipper",
    ),
    _t2(
        "t2_hs_invoice_pl_mismatch",
        [
            {
                "type": "invoice",
                "invoice_number": "INV-XM-889",
                "goods_description": "Stainless steel kitchen sinks",
                "quantity": 400,
                "declared_value_usd": 32000,
            },
            {
                "type": "packing_list",
                "goods_description": "Ceramic bathroom sinks",
                "quantity": 400,
            },
            {
                "type": "bill_of_lading",
                "bl_number": "OOLU9988776655",
                "invoice_number": "INV-XM-889",
                "port_of_loading": "CNXMN",
                "port_of_discharge": "INHZA",
            },
        ],
        ["goods_description_mismatch_invoice_vs_packing_list"],
        "hold",
    ),
    _t2(
        "t2_missing_origin",
        [
            {
                "type": "invoice",
                "invoice_number": "INV-NB-441",
                "goods_description": "Plastic storage crates",
                "quantity": 2000,
                "declared_value_usd": 9000,
            },
            {
                "type": "packing_list",
                "quantity": 2000,
            },
            {
                "type": "bill_of_lading",
                "bl_number": "MSCU5544332211",
                "invoice_number": "INV-NB-441",
                "port_of_loading": "CNNBO",
                "port_of_discharge": "INMAA",
            },
        ],
        ["missing_country_of_origin"],
        "hold",
    ),
    _t2(
        "t2_weight_mismatch",
        [
            {
                "type": "invoice",
                "invoice_number": "INV-TJ-102",
                "goods_description": "Cast iron cookware sets",
                "quantity": 800,
                "declared_value_usd": 24000,
                "country_of_origin": "China",
            },
            {
                "type": "packing_list",
                "quantity": 800,
                "gross_weight_kg": 9600,
            },
            {
                "type": "bill_of_lading",
                "bl_number": "EGLV9988123",
                "invoice_number": "INV-TJ-102",
                "gross_weight_kg": 8200,
                "port_of_loading": "CNTXG",
                "port_of_discharge": "INNSA",
            },
        ],
        ["weight_mismatch_packing_vs_bl"],
        "query_shipper",
    ),
    _t2(
        "t2_dual_doc_error",
        [
            {
                "type": "invoice",
                "invoice_number": "INV-GZ-600",
                "goods_description": "Office chairs with metal frames",
                "quantity": 300,
                "declared_value_usd": 9000,
                "country_of_origin": "China",
            },
            {
                "type": "packing_list",
                "quantity": 280,
            },
            {
                "type": "bill_of_lading",
                "bl_number": "COSU8877665544",
                "invoice_number": "INV-GZ-601",
                "port_of_loading": "CNGGZ",
                "port_of_discharge": "INBLR",
            },
        ],
        ["quantity_mismatch", "invoice_number_mismatch_bl_vs_invoice"],
        "hold",
    ),
    _t2(
        "t2_currency_discharge",
        [
            {
                "type": "invoice",
                "invoice_number": "INV-SH-7712",
                "goods_description": "Laptop cooling pads",
                "quantity": 5000,
                "declared_value_usd": 15000,
                "currency": "USD",
                "country_of_origin": "China",
            },
            {
                "type": "packing_list",
                "quantity": 5000,
            },
            {
                "type": "bill_of_lading",
                "bl_number": "MAEU1122334455",
                "invoice_number": "INV-SH-7712",
                "port_of_discharge": "INMAA",
                "freight_prepaid": True,
                "notify_party": "",
            },
        ],
        ["missing_notify_party", "suspected_undervaluation"],
        "query_shipper",
    ),
    _t2(
        "t2_consignee_typo",
        [
            {
                "type": "invoice",
                "invoice_number": "INV-DL-909",
                "consignee": "Nova Electronics Pvt Ltd, New Delhi",
                "goods_description": "Ethernet patch cords Cat6",
                "quantity": 8000,
                "declared_value_usd": 12000,
                "country_of_origin": "China",
            },
            {
                "type": "packing_list",
                "quantity": 8000,
            },
            {
                "type": "bill_of_lading",
                "bl_number": "CMDU4455667788",
                "invoice_number": "INV-DL-909",
                "consignee": "Nova Electonics Pvt Ltd, New Delhi",
                "port_of_loading": "CNSHA",
                "port_of_discharge": "INDRI",
            },
        ],
        ["consignee_name_mismatch"],
        "hold",
    ),
]

# --- Task 3: messy multi-doc clearance ---

TASK3_DOCUMENTS: list[dict[str, Any]] = [
    _t3(
        "t3_electronic_accessories",
        {
            "documents": [
                {
                    "type": "invoice",
                    "invoice_number": "INV-SZ-MIX-01",
                    "goods_description": "Electronic accessories (mixed)",
                    "quantity": 2000,
                    "declared_value_usd": 4000,
                    "country_of_origin": "China",
                    "lines": [
                        {"desc": "Wireless earbuds", "qty": 800},
                        {"desc": "USB hubs", "qty": 600},
                        {"desc": "Phone cases", "qty": 600},
                    ],
                },
                {
                    "type": "packing_list",
                    "quantity": 1950,
                    "notes": "Partial shipment as per shipper",
                },
                {
                    "type": "bill_of_lading",
                    "bl_number": "OOLU5566778899",
                    "invoice_number": "INV-SZ-MIX-01",
                    "port_of_loading": "KRPUS",
                    "port_of_discharge": "INNSA",
                    "shipper_remarks": "Freight collect",
                },
                {
                    "type": "certificate_of_origin",
                    "issuer": "Shenzhen Chamber",
                    "country_of_origin": "China",
                    "partial_chinese": "货物名称: 电子配件",
                },
            ]
        },
        "8518.30.00",
        [
            "quantity_mismatch",
            "vague_goods_description",
            "origin_loading_mismatch",
            "suspected_undervaluation",
            "mixed_consignment_requires_classification",
        ],
        "hold",
        assessable_inr=3_600_000.0,
        duty_inr=720_000.0,
    ),
    _t3(
        "t3_chemical_dual_use",
        {
            "documents": [
                {
                    "type": "invoice",
                    "invoice_number": "INV-SH-CHEM-22",
                    "goods_description": "Organic chemical intermediate for industrial synthesis",
                    "quantity": 2000,
                    "unit": "kg",
                    "declared_value_usd": 45000,
                    "country_of_origin": "Germany",
                },
                {
                    "type": "packing_list",
                    "quantity_kg": 2000,
                },
                {
                    "type": "bill_of_lading",
                    "bl_number": "HLCU9988776655",
                    "invoice_number": "INV-SH-CHEM-22",
                    "port_of_loading": "DEHAM",
                    "port_of_discharge": "INMAA",
                },
                {
                    "type": "msds",
                    "substance": "Anthranilic acid derivatives",
                    "hazard_class": "irritant",
                },
            ]
        },
        "2921.41.00",
        [
            "dual_use_or_controlled_chemical_risk",
            "msds_present_requires_review",
        ],
        "refer_to_customs",
        assessable_inr=3_750_000.0,
        duty_inr=562_500.0,
    ),
    _t3(
        "t3_underinvoice_mixed_hs",
        {
            "documents": [
                {
                    "type": "invoice",
                    "invoice_number": "INV-XMN-778",
                    "goods_description": "Electronic accessories",
                    "declared_value_usd": 800,
                    "quantity": 500,
                    "country_of_origin": "China",
                },
                {
                    "type": "packing_list",
                    "quantity": 520,
                },
                {
                    "type": "bill_of_lading",
                    "bl_number": "MSCU3344556677",
                    "invoice_number": "INV-XMN-778",
                    "port_of_loading": "CNXMN",
                    "port_of_discharge": "INHZA",
                },
            ]
        },
        "8544.42.90",
        [
            "quantity_mismatch",
            "suspected_undervaluation",
            "vague_goods_description",
        ],
        "query_shipper",
        assessable_inr=720_000.0,
        duty_inr=108_000.0,
    ),
    _t3(
        "t3_machinery_vague",
        {
            "documents": [
                {
                    "type": "invoice",
                    "invoice_number": "INV-QD-MECH-03",
                    "goods_description": "Industrial machine parts and accessories",
                    "declared_value_usd": 95000,
                    "quantity": 4,
                    "country_of_origin": "Japan",
                },
                {
                    "type": "packing_list",
                    "quantity": 4,
                    "net_weight_kg": 4200,
                },
                {
                    "type": "bill_of_lading",
                    "bl_number": "ONEY8877665544",
                    "invoice_number": "INV-QD-MECH-03",
                    "port_of_loading": "JPYOK",
                    "port_of_discharge": "INPNQ",
                },
            ]
        },
        "8479.89.90",
        [
            "vague_goods_description",
            "high_value_shipment",
        ],
        "hold",
        assessable_inr=7_900_000.0,
        duty_inr=1_185_000.0,
    ),
    _t3(
        "t3_textile_origin",
        {
            "documents": [
                {
                    "type": "invoice",
                    "invoice_number": "INV-BD-TEX-14",
                    "goods_description": "Cotton knitted apparel",
                    "declared_value_usd": 22000,
                    "quantity": 3000,
                    "country_of_origin": "Bangladesh",
                },
                {
                    "type": "packing_list",
                    "quantity": 3000,
                },
                {
                    "type": "bill_of_lading",
                    "bl_number": "MAEU2233445566",
                    "invoice_number": "INV-BD-TEX-14",
                    "port_of_loading": "CNSHA",
                    "port_of_discharge": "INNSA",
                },
            ]
        },
        "6109.10.00",
        [
            "origin_loading_mismatch",
            "textile_declaration_review",
        ],
        "hold",
        assessable_inr=1_830_000.0,
        duty_inr=640_500.0,
    ),
    _t3(
        "t3_partial_chinese_invoice",
        {
            "documents": [
                {
                    "type": "invoice",
                    "invoice_number": "SZ-电子-991",
                    "goods_description": "Bluetooth audio devices and accessories",
                    "declared_value_usd": 28000,
                    "quantity": 1400,
                    "country_of_origin": "China",
                    "language_notes": "Line items partially in Chinese on original PDF",
                },
                {
                    "type": "packing_list",
                    "quantity": 1350,
                },
                {
                    "type": "bill_of_lading",
                    "bl_number": "EGLV6655443322",
                    "invoice_number": "SZ-991-ENG",
                    "port_of_loading": "CNSZX",
                    "port_of_discharge": "INMAA",
                },
            ]
        },
        "8518.30.00",
        [
            "quantity_mismatch",
            "invoice_number_mismatch_bl_vs_invoice",
            "multilingual_document_review",
        ],
        "hold",
        assessable_inr=2_330_000.0,
        duty_inr=466_000.0,
    ),
    _t3(
        "t3_clear_simple",
        {
            "documents": [
                {
                    "type": "invoice",
                    "invoice_number": "INV-TJ-STD-01",
                    "goods_description": "USB 3.0 cables 1m length",
                    "quantity": 20000,
                    "declared_value_usd": 30000,
                    "country_of_origin": "China",
                },
                {
                    "type": "packing_list",
                    "quantity": 20000,
                },
                {
                    "type": "bill_of_lading",
                    "bl_number": "CMDU7788990011",
                    "invoice_number": "INV-TJ-STD-01",
                    "port_of_loading": "CNTXG",
                    "port_of_discharge": "INNSA",
                },
            ]
        },
        "8544.42.90",
        [],
        "clear",
        assessable_inr=2_490_000.0,
        duty_inr=373_500.0,
    ),
    _t3(
        "t3_risk_refer",
        {
            "documents": [
                {
                    "type": "invoice",
                    "invoice_number": "INV-OBSCURE-01",
                    "goods_description": "Laboratory reagents (details withheld pending end-user)",
                    "declared_value_usd": 120000,
                    "quantity": 50,
                    "country_of_origin": "USA",
                },
                {
                    "type": "packing_list",
                    "quantity": 50,
                },
                {
                    "type": "bill_of_lading",
                    "bl_number": "HLCU4455667788",
                    "invoice_number": "INV-OBSCURE-01",
                    "port_of_loading": "USNYC",
                    "port_of_discharge": "INMAA",
                },
            ]
        },
        "2921.41.00",
        [
            "ambiguous_end_use",
            "high_value_chemical_shipment",
        ],
        "refer_to_customs",
        assessable_inr=9_960_000.0,
        duty_inr=1_494_000.0,
    ),
]

TASK_DOCUMENTS: dict[str, list[dict[str, Any]]] = {
    "task1": TASK1_DOCUMENTS,
    "task2": TASK2_DOCUMENTS,
    "task3": TASK3_DOCUMENTS,
}


def get_document_by_task(task_id: str, index: int) -> dict[str, Any]:
    docs = TASK_DOCUMENTS[task_id]
    return docs[index % len(docs)]


def list_task_ids() -> list[str]:
    return list(TASK_DOCUMENTS.keys())


def get_shipment_by_id(shipment_id: str) -> dict[str, Any] | None:
    for _tid, docs in TASK_DOCUMENTS.items():
        for d in docs:
            if d.get("id") == shipment_id:
                return d
    return None


def find_task_for_shipment(shipment_id: str) -> str | None:
    for tid, docs in TASK_DOCUMENTS.items():
        for d in docs:
            if d.get("id") == shipment_id:
                return tid
    return None
