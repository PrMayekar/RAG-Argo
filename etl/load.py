from db.connection import get_session
from db.models import ArgoFloat, Profile, Measurement

def load(transformed_data: dict):
    session = get_session()
    try:
        fi = transformed_data["float_info"]

        argo_float = session.query(ArgoFloat).filter_by(wmo_id=fi["wmo_id"]).first()
        if not argo_float:
            argo_float = ArgoFloat(**fi)
            session.add(argo_float)
            session.flush()
            print(f"[LOAD] New float inserted: {fi['wmo_id']}")
        else:
            print(f"[LOAD] Float {fi['wmo_id']} already exists, skipping insert.")

        inserted_profiles     = 0
        inserted_measurements = 0

        for prof in transformed_data["profiles"]:
            measurements = prof.pop("measurements")

            exists = session.query(Profile).filter_by(
                float_id=argo_float.id,
                cycle_number=prof["cycle_number"]
            ).first()
            if exists:
                continue

            profile = Profile(float_id=argo_float.id, **prof)
            session.add(profile)
            session.flush()

            session.bulk_insert_mappings(Measurement, [
                {"profile_id": profile.id, **m} for m in measurements
            ])

            inserted_profiles     += 1
            inserted_measurements += len(measurements)

        session.commit()
        print(f"[LOAD] {inserted_profiles} profiles | {inserted_measurements} measurements inserted.")

    except Exception as e:
        session.rollback()
        print(f"[LOAD] Error: {e}")
        raise
    finally:
        session.close()