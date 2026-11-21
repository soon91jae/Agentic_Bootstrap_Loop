import argparse, json
from pathlib import Path
from .readers import READERS
from .schema import Record, read_jsonl, write_json
from .splits import split_records, save_dataset, validate_splits
from .teacher import GPTTeacher, OracleTeacher
from .loop import run
from .models import LinearClassifier
from .metrics import evaluate


def load_records(directory):
    return [Record(**r) for s in ("train","dev","test") for r in read_jsonl(Path(directory)/f"{s}.jsonl")]


def main():
    parser=argparse.ArgumentParser()
    sub=parser.add_subparsers(dest="command",required=True)
    prep=sub.add_parser("prepare")
    prep.add_argument("--reader",choices=READERS,required=True)
    prep.add_argument("--input",required=True); prep.add_argument("--mapping")
    prep.add_argument("--definitions",required=True); prep.add_argument("--output",required=True)
    prep.add_argument("--seed",type=int,default=42); prep.add_argument("--keep-splits",action="store_true")
    train=sub.add_parser("run")
    train.add_argument("--data",required=True); train.add_argument("--config",required=True)
    train.add_argument("--output",required=True)
    train.add_argument("--oracle-simulation",action="store_true",help="Use TRAIN gold as a perfect teacher; NOT GPT")
    ev=sub.add_parser("evaluate")
    ev.add_argument("--data",required=True); ev.add_argument("--run",required=True)
    ev.add_argument("--split",choices=["dev","test"],default="test")
    dg=sub.add_parser("diagnose")
    dg.add_argument("--data",required=True); dg.add_argument("--run",required=True)
    dg.add_argument("--split",choices=["dev","test"],default="dev")
    dg.add_argument("--ks",type=int,nargs="+",default=[10,25,50,100])
    args=parser.parse_args()
    if args.command=="prepare":
        mapping=json.loads(Path(args.mapping).read_text()) if args.mapping else None
        if args.reader!="odd" and mapping is None: parser.error("This reader requires --mapping")
        output=Path(args.output)
        if output.exists() and any(output.iterdir()): parser.error("Prepare output must be empty")
        records=READERS[args.reader]().read(args.input,mapping)
        definitions=json.loads(Path(args.definitions).read_text())
        if not definitions or any(not isinstance(d,str) or not d.strip() for d in definitions.values()):
            parser.error("Definitions must be a nonempty code-to-text mapping")
        if any(set(r.labels)-set(definitions) for r in records): parser.error("Labels missing from definitions")
        validate_splits(records) if args.keep_splits else split_records(records,args.seed)
        print(json.dumps(save_dataset(records,definitions,args.output),indent=2))
    elif args.command=="run":
        records=load_records(args.data); validate_splits(records)
        definitions=json.loads((Path(args.data)/"definitions.json").read_text())
        config=json.loads(Path(args.config).read_text())
        if args.oracle_simulation:
            print("ORACLE SIMULATION: training gold labels are used as teacher answers")
            teacher=OracleTeacher([r for r in records if r.split=="train"])
            config["teacher_provenance"]="oracle_simulation"
        else:
            if "teacher" not in config: parser.error("Provide teacher config, or explicit --oracle-simulation")
            teacher=GPTTeacher(**config["teacher"])
            config["teacher_provenance"]="gpt_silver"
        run(config,records,definitions,teacher,args.output)
    else:
        definitions=json.loads((Path(args.run)/"definitions.json").read_text())
        cfg=json.loads((Path(args.run)/"config.json").read_text())
        if cfg.get("classifier",{}).get("backend")=="hf":
            from .hf import HFClassifier
            classifier=HFClassifier.load(Path(args.run)/"best_classifier",definitions)
        else:
            state=json.loads((Path(args.run)/"best_classifier.json").read_text())
            classifier=LinearClassifier(definitions,state["dim"],state["lr"])
            classifier.w=state["w"]; classifier.b=state["b"]
        records=load_records(args.data); validate_splits(records)
        if args.command=="diagnose":
            from .diagnostics import diagnose, load_retriever
            from .models import TrainableRetriever
            retriever=load_retriever(Path(args.run)/"best_retriever.json")
            initial=TrainableRetriever(definitions,**cfg.get("retriever",{}))
            chosen=[r for r in records if r.split==args.split]
            result={"split":args.split,"teacher_provenance":cfg.get("teacher_provenance"),
                    "checkpoint":json.loads((Path(args.run)/"best.json").read_text()),
                    "trained":diagnose(retriever,classifier,chosen,definitions,args.ks),
                    "initial_retriever":diagnose(initial,classifier,chosen,definitions,args.ks)}
            write_json(Path(args.run)/f"diagnostics_{args.split}.json",result)
            print(json.dumps(result,indent=2))
            return
        result=evaluate(classifier,[r for r in records if r.split==args.split],definitions)
        write_json(Path(args.run)/f"final_{args.split}_metrics.json",result)
        print(json.dumps(result,indent=2))

if __name__=="__main__": main()
