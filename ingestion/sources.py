"""Real, messy source documents to ingest — official docs pages (HTML) and academic papers (PDF)
on the same data-engineering topics already covered by data/corpus.json's hand-written chunks."""

from dataclasses import dataclass


@dataclass
class Source:
    id: str
    topic: str
    doc_type: str  # "html" or "pdf"
    url: str
    title: str


SOURCES: list[Source] = [
    Source(
        id="spark_docs",
        topic="spark",
        doc_type="html",
        url="https://spark.apache.org/docs/latest/rdd-programming-guide.html",
        title="Spark RDD Programming Guide",
    ),
    Source(
        id="airflow_docs",
        topic="airflow",
        doc_type="html",
        url="https://airflow.apache.org/docs/apache-airflow/stable/core-concepts/dags.html",
        title="Airflow DAGs",
    ),
    Source(
        id="dbt_docs",
        topic="dbt",
        doc_type="html",
        url="https://docs.getdbt.com/docs/build/sql-models",
        title="dbt SQL Models",
    ),
    Source(
        id="flink_docs",
        topic="flink",
        doc_type="html",
        url="https://nightlies.apache.org/flink/flink-docs-stable/docs/concepts/time/",
        title="Flink Event Time and Watermarks",
    ),
    Source(
        id="snowflake_docs",
        topic="snowflake",
        doc_type="html",
        url="https://docs.snowflake.com/en/user-guide/tables-clustering-micropartitions",
        title="Snowflake Micro-partitions and Clustering",
    ),
    Source(
        id="spark_paper",
        topic="spark",
        doc_type="pdf",
        url="https://www.usenix.org/system/files/conference/nsdi12/nsdi12-final138.pdf",
        title="Resilient Distributed Datasets (Zaharia et al., NSDI 2012)",
    ),
    Source(
        id="delta_lake_paper",
        topic="delta_lake",
        doc_type="pdf",
        url="https://www.vldb.org/pvldb/vol13/p3411-armbrust.pdf",
        title="Delta Lake: High-Performance ACID Table Storage (Armbrust et al., VLDB 2020)",
    ),
    # Genuinely scanned/image-only PDFs (no embedded text layer at all) — public OCR test
    # fixtures from the OCRmyPDF project (MIT licensed), used here to exercise the OCR fallback
    # path rather than for their (off-topic) content.
    Source(
        id="ocr_demo_brochure",
        topic="ocr_demo",
        doc_type="pdf",
        url="https://raw.githubusercontent.com/ocrmypdf/OCRmyPDF/main/tests/resources/skew.pdf",
        title="Scanned vintage product brochure, slightly skewed (OCRmyPDF test fixture)",
    ),
    Source(
        id="ocr_demo_french",
        topic="ocr_demo",
        doc_type="pdf",
        url="https://raw.githubusercontent.com/ocrmypdf/OCRmyPDF/main/tests/resources/francais.pdf",
        title="Scanned French text, non-English OCR test (OCRmyPDF test fixture)",
    ),
]
