"""Audited, lossless schema correction after the first final replay failed.

Does not modify frozen research code or models. Preserves the original prefix
and checks every timestamp before replacing the canonical data directory.
"""
import datetime,hashlib,json,shutil,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from otg.protocol import verify_final_protocol
ROOT=Path(__file__).resolve().parents[1]


def main():
    _,digest=verify_final_protocol(ROOT)
    parent=ROOT/'data/processed';old=parent/'borg_prefix_29d';out=parent/'borg_prefix_29d_corrected_staging';out.mkdir(exist_ok=False)
    metadata=json.loads((old/'metadata.json').read_text());audits=[]
    for name in ['events.parquet','intervals.parquet']:
        source=old/name;reader=pq.ParquetFile(source);schema=reader.schema_arrow;column=schema.get_field_index('time_us')
        if not pa.types.is_float64(schema.field(column).type):raise ValueError('Unexpected schema; do not apply this correction')
        fields=list(schema);fields[column]=pa.field('time_us',pa.int64(),nullable=False);md=dict(schema.metadata or {})
        if b'pandas' in md:
            pm=json.loads(md[b'pandas'])
            for col in pm['columns']:
                if col['field_name']=='time_us':col.update(pandas_type='int64',numpy_type='int64')
            md[b'pandas']=json.dumps(pm).encode()
        corrected=pa.schema(fields,metadata=md);rows=0;minimum=None;maximum=None
        with pq.ParquetWriter(out/name,corrected,compression='zstd') as writer:
            for batch in reader.iter_batches(batch_size=1_000_000):
                time=batch.column(column)
                if time.null_count or not pc.all(pc.is_finite(time)).as_py():raise ValueError('Null/nonfinite timestamps')
                integer=pc.cast(time,pa.int64(),safe=True)
                if not pc.all(pc.equal(time,pc.cast(integer,pa.float64()))).as_py():raise ValueError('Non-lossless cast')
                lo=pc.min(integer).as_py();hi=pc.max(integer).as_py()
                if lo<0 or hi>=metadata['stop_us']:raise ValueError('Timestamp outside supported prefix')
                minimum=lo if minimum is None else min(minimum,lo);maximum=hi if maximum is None else max(maximum,hi)
                arrays=list(batch.columns);arrays[column]=integer
                writer.write_batch(pa.RecordBatch.from_arrays(arrays,schema=corrected));rows+=batch.num_rows
        if rows!=reader.metadata.num_rows:raise ValueError('Changed row count')
        audits.append({'file':name,'rows_checked':rows,'minimum':minimum,'maximum':maximum,'old_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'new_sha256':hashlib.sha256((out/name).read_bytes()).hexdigest(),'all_values_exactly_preserved':True})
        print(json.dumps(audits[-1]),flush=True)
    for name in ['task_index.parquet','job_index.parquet','machine_index.parquet']:shutil.copy2(old/name,out/name)
    report={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'protocol_sha256':digest,'reason':'Final cohort merge rejected float64 parquet timestamps against int64 cutoff queries. No Borg final cohort score existed. Original data and failed logs preserved. Exact origin of the dtype promotion not established.','method':'Safe float64-to-int64 cast with every-row finite/null/range and roundtrip equality checks; all other columns copied unchanged.','audits':audits,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'original_metadata_sha256':hashlib.sha256((old/'metadata.json').read_bytes()).hexdigest(),'models_or_evaluation_choices_changed':False}
    metadata.update(events_sha256=audits[0]['new_sha256'],intervals_sha256=audits[1]['new_sha256'],timestamp_schema_correction=report)
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    old.rename(parent/'borg_prefix_29d_original_float_schema');out.rename(old)
    failed=parent/'borg_final_cohorts_v1'
    if failed.exists():
        if list(failed.iterdir()):raise RuntimeError('Unexpected partial cohort outputs; preserve and inspect manually')
        failed.rename(parent/'borg_final_cohorts_v1_empty_failed_attempt')
    (ROOT/'reports/final_borg_timestamp_correction.json').write_text(json.dumps(report,indent=2)+'\n')
    with (ROOT/'LEDGER.md').open('a') as f:f.write('\n## Final Borg timestamp-schema correction\n\nThe original final prefix stored time_us as float64; frozen replay rejected its join to int64 origins before any cohort output/scoring. A separately recorded lossless conversion checked every timestamp and preserved all original artifacts. Models, frozen code, periods, labels, thresholds and comparisons are unchanged. See reports/final_borg_timestamp_correction.json.\n')
    print(json.dumps({'status':'corrected','protocol_unchanged':digest}),flush=True)


if __name__=='__main__':main()
