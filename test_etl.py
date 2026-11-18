#test_etl.py
import psycopg2
import os
from dotenv import load_dotenv
load_dotenv()

conn = psycopg2.connect(
    host=os.getenv("DB_HOST"), port=os.getenv("DB_PORT"),
    dbname=os.getenv("DB_NAME"), user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD")
)
cur = conn.cursor()

def test_tables_not_empty():
    for table in ["argo_floats", "profiles", "measurements"]:
        cur.execute(f"SELECT COUNT(*) FROM {table}")
        count = cur.fetchone()[0]
        assert count > 0, f"FAIL: {table} is empty!"
        print(f"PASS: {table} has {count} rows")

def test_no_null_wmo_ids():
    cur.execute("SELECT COUNT(*) FROM argo_floats WHERE wmo_id IS NULL")
    nulls = cur.fetchone()[0]
    assert nulls == 0, f"FAIL: {nulls} floats have NULL wmo_id!"
    print("PASS: All floats have wmo_id")

def test_no_orphan_profiles():
    cur.execute("""
        SELECT COUNT(*) FROM profiles p
        LEFT JOIN argo_floats f ON f.id = p.float_id
        WHERE f.id IS NULL
    """)
    orphans = cur.fetchone()[0]
    assert orphans == 0, f"FAIL: {orphans} profiles have no parent float!"
    print("PASS: No orphaned profiles")

def test_no_orphan_measurements():
    cur.execute("""
        SELECT COUNT(*) FROM measurements m
        LEFT JOIN profiles p ON p.id = m.profile_id
        WHERE p.id IS NULL
    """)
    orphans = cur.fetchone()[0]
    assert orphans == 0, f"FAIL: {orphans} measurements have no parent profile!"
    print("PASS: No orphaned measurements")

def test_temperature_range():
    cur.execute("""
        SELECT MIN(temperature), MAX(temperature)
        FROM measurements
        WHERE temp_qc = '1'
    """)
    mn, mx = cur.fetchone()
    assert -3 <= mn, f"FAIL: Min temperature {mn}°C is impossibly cold!"
    assert mx <= 35,  f"FAIL: Max temperature {mx}°C is impossibly hot!"
    print(f"PASS: Temperature range {mn:.2f}°C to {mx:.2f}°C — realistic")

def test_coordinates_valid():
    cur.execute("""
        SELECT COUNT(*) FROM profiles
        WHERE latitude < -90 OR latitude > 90
        OR longitude < -180 OR longitude > 180
    """)
    bad = cur.fetchone()[0]
    assert bad == 0, f"FAIL: {bad} profiles have invalid coordinates!"
    print("PASS: All coordinates are valid")

def test_qc_flags_valid():
    cur.execute("""
        SELECT COUNT(*) FROM measurements
        WHERE temp_qc NOT IN ('1','2','3','4','8','9')
        AND temp_qc IS NOT NULL
    """)
    bad = cur.fetchone()[0]
    assert bad == 0, f"FAIL: {bad} measurements have invalid QC flags!"
    print("PASS: All QC flags are valid")

if __name__ == "__main__":
    print("\n===== ETL DATA QUALITY TESTS =====\n")
    tests = [
        test_tables_not_empty,
        test_no_null_wmo_ids,
        test_no_orphan_profiles,
        test_no_orphan_measurements,
        test_temperature_range,
        test_coordinates_valid,
        test_qc_flags_valid,
    ]
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except AssertionError as e:
            print(f"  {e}")
            failed += 1
        except Exception as e:
            print(f"ERROR in {t.__name__}: {e}")
            failed += 1

    print(f"\n{'='*35}")
    print(f"Results: {passed} passed, {failed} failed")
    cur.close()
    conn.close()
