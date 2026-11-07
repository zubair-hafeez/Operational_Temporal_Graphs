"""Check that acquiring later records did not alter any earlier feature prefix."""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from extension.prepare_pilots import load

def main():
    results = {}
    for source in ["uci", "bpi2014", "azure_full"]:
        old, cats, nums, rel, bounds, *_ = load(source, evaluation=False)
        new, nc, nn, nr, nb, *_ = load(source, evaluation=True)
        assert (cats, nums, rel) == (nc, nn, nr)
        np.testing.assert_array_equal(bounds, nb)
        new = new[new.time < bounds[-1]].reset_index(drop=True)
        keys = list(dict.fromkeys(["id", "time", "start", "state", "event_count"] + cats + nums))
        pd.testing.assert_frame_equal(old[keys], new[keys])
        # IDs may be globally recoded by future categories; only equality matters.
        for relation in rel:
            np.testing.assert_array_equal(old[relation].to_numpy()<0, new[relation].to_numpy()<0)
            np.testing.assert_array_equal(pd.factorize(old[relation])[0], pd.factorize(new[relation])[0])
        encoders = [OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=5).fit(f.loc[f.time < bounds[1], cats]) for f in [old, new]]
        for a, b in zip(encoders[0].categories_, encoders[1].categories_): np.testing.assert_array_equal(a,b)
        np.testing.assert_array_equal(encoders[0].transform(old[cats]), encoders[1].transform(new[cats]))
        results[source] = {"historical_rows": len(old), "all_observed_attributes_identical": True, "relation_partition_and_missing_mask_identical": True, "training_vocabulary_and_encodings_identical": True, "future_end_labels_excluded_from_comparison": True}
        print(json.dumps({source:results[source]}), flush=True)
    (ROOT/"reports/extension_20261021/prefix_extension_audit.json").write_text(json.dumps({"status":"passed", "sources":results}, indent=2)+"\n")

if __name__ == "__main__": main()
