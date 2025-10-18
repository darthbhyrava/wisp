#!/usr/bin/env python3
import argparse
import json
import numpy as np
import sys

from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any


class WISPBenchScorer:
    def __init__(self, jsonl_file: str):
        self.raw_data = self._load_data(jsonl_file)
        self.annotator_counts = self._compute_annotator_counts()
        if self._is_arbitrated_data():
            self.data = self._filter_arbitrated_data()
        else:
            self.data = self._filter_annotator_sessions('harsh')
        self.methods = self._extract_linearization_methods()
        self.all_options = self._extract_all_options()
        self.all_poems = self._extract_all_poems()
        self.wisp_label_mapping = self._build_wisp_label_mapping()

    def _compute_annotator_counts(self) -> Dict[str, int]:
        counts = Counter()
        for ann in self.raw_data:
            sid = ann.get('_session_id')
            if not sid:
                continue
            annot = sid.split('-')[-1]
            counts[annot] += 1
        return dict(counts)
        
    def _load_data(self, jsonl_file: str) -> List[Dict]:
        data = []
        try:
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if line.strip():
                        try:
                            data.append(json.loads(line))
                        except json.JSONDecodeError as e:
                            print(f"Warning: Skipping malformed JSON on line {line_num}: {e}")
            print(f"Loaded {len(data)} total annotations from {jsonl_file}")
            return data
        except FileNotFoundError:
            print(f"Error: File {jsonl_file} not found")
            sys.exit(1)

    def _is_arbitrated_data(self) -> bool:
        """Check if this is arbitrated data by looking for _arbitrated field."""
        return any(ann.get('_arbitrated', False) for ann in self.raw_data[:10])  # Check first 10 entries
    
    def _filter_arbitrated_data(self) -> List[Dict]:
        """Filter arbitrated data to include harsh sessions."""
        filtered_data = []
        for ann in self.raw_data:
            session_id = ann.get('_session_id', '')
            # Include harsh sessions (for non-trafilatura methods)
            if session_id.endswith('harsh'):
                filtered_data.append(ann)
        print(f"Filtered arbitrated data to {len(filtered_data)} annotations (harsh sessions only)")
        return filtered_data
        
    def _filter_annotator_sessions(self, suffix: str) -> List[Dict]:
        filtered_data = [ann for ann in self.raw_data if ann.get('_session_id', '').endswith(suffix)]
        print(f"Filtered to {len(filtered_data)} annotations with '{suffix}' sessions")
        return filtered_data

    def _extract_linearization_methods(self) -> Set[str]:
        methods = set()
        for ann in self.data:
            if 'meta' in ann and 'method' in ann['meta']:
                methods.add(ann['meta']['method'])
            elif 'method' in ann:
                methods.add(ann['method'])
            elif 'meta' in ann and 'category' in ann['meta']:
                methods.add(ann['meta']['category'])
        return methods
    
    def _extract_all_options(self) -> Set[str]:
        options = set()
        for ann in self.data:
            for option in ann.get('options', []):
                options.add(option['id'])
        return options
    
    def _extract_all_poems(self) -> Set[str]:
        return {ann['filename'] for ann in self.data}
    
    def _build_wisp_label_mapping(self) -> Dict[str, Set[str]]:
        mapping = defaultdict(set)
        for option in self.all_options:
            if option.startswith('line_breaks'):
                mapping['LINE_BREAKS'].add(option)
            elif option.startswith('internal'):
                mapping['INTERNAL'].add(option)
            elif option.startswith('prefix'):
                mapping['PREFIX'].add(option)
            elif option.startswith('vertical'):
                mapping['VERTICAL'].add(option)
            elif option.startswith('ocr_error'):
                mapping['OCR_ERROR'].add(option)
            else:
                mapping['OTHER'].add(option)
        return dict(mapping)
    
    def _get_method_for_annotation(self, ann: Dict) -> str:
        if 'meta' in ann and 'method' in ann['meta']:
            return ann['meta']['method']
        elif 'method' in ann:
            return ann['method']
        elif 'meta' in ann and 'category' in ann['meta']:
            return ann['meta']['category']
        return ""
    
    def calculate_method_scores(self) -> Dict[str, Dict]:
        """Calculate method scores with OCR catastrophic filtering and reliability metrics.

        Definitions per annotation:
          catastrophic: any ocr_error* accepted AND no non-OCR accepts
          mixed:        ocr_error* accepted AND ≥1 non-OCR accept
          pure:         no ocr_error* accepted (accepts may be empty or non-OCR)

        CORRECTED: macro_score / weighted_macro_score now include catastrophic annotations as 0% performance.
        non_ocr_macro_score / non_ocr_weighted_macro_score are computed on pure annotations only.
        composite_score = macro_score * (1 - (cat_rate + 0.5 * mixed_rate)).
        """
        method_scores = {}
        for method in self.methods:
            method_data = [ann for ann in self.data if self._get_method_for_annotation(ann) == method]
            if not method_data:
                continue
            # setup catastrophic/mixed/pure splits
            catastrophic_annotations = []
            mixed_annotations = []
            pure_annotations = []
            for ann in method_data:
                accepts = ann.get('accept', []) or []
                ocr_accept = any(a.startswith('ocr_error') for a in accepts)
                non_ocr_accepts = [a for a in accepts if not a.startswith('ocr_error')]
                if ocr_accept and not non_ocr_accepts:
                    catastrophic_annotations.append(ann)
                elif ocr_accept and non_ocr_accepts:
                    mixed_annotations.append(ann)
                else:
                    pure_annotations.append(ann)
            filtered_annotations = mixed_annotations + pure_annotations  # excluded: catastrophic
            all_annotations = catastrophic_annotations + mixed_annotations + pure_annotations  # include all for macro score

            option_scores: Dict[str, Dict[str, Any]] = {}
            weighted_true_sum = 0
            weighted_total_sum = 0
            pure_weighted_true_sum = 0
            pure_weighted_total_sum = 0
            # For true macro score including catastrophic as 0%
            all_weighted_true_sum = 0
            all_weighted_total_sum = 0

            for option in self.all_options:
                # original availability (before catastrophic filtering)
                original_available = [ann for ann in method_data if any(opt['id'] == option for opt in ann.get('options', []))]
                if not original_available:
                    continue
                original_total = len(original_available)
                original_true = sum(1 for ann in original_available if option in ann.get('accept', []))

                # filtered availability (exclude catastrophic) - for legacy metrics
                filtered_available = [ann for ann in filtered_annotations if any(opt['id'] == option for opt in ann.get('options', []))]
                filtered_total = len(filtered_available)
                filtered_true = sum(1 for ann in filtered_available if option in ann.get('accept', []))
                
                # true availability (include catastrophic as 0%) - for corrected macro score
                all_available = [ann for ann in all_annotations if any(opt['id'] == option for opt in ann.get('options', []))]
                all_total = len(all_available)
                all_true = sum(1 for ann in all_available if option in ann.get('accept', []) and ann not in catastrophic_annotations)
                if filtered_total > 0:
                    pass_rate = (filtered_true / filtered_total * 100)
                else:
                    pass_rate = 0
                    
                # true pass rate including catastrophic as 0%
                if all_total > 0:
                    true_pass_rate = (all_true / all_total * 100)
                else:
                    true_pass_rate = 0

                # pure-only availability (for non_ocr* metrics)
                pure_available = [ann for ann in pure_annotations if any(opt['id'] == option for opt in ann.get('options', []))]
                pure_total = len(pure_available)
                pure_true = sum(1 for ann in pure_available if option in ann.get('accept', []))
                if pure_total > 0:
                    pure_pass_rate = (pure_true / pure_total * 100)
                else:
                    pure_pass_rate = 0

                option_scores[option] = {
                    'pass_rate': pass_rate,  # legacy filtered rate (for backward compatibility)
                    'true_count': filtered_true,
                    'total_count': filtered_total,
                    'original_total_count': original_total,
                    'coverage_shrink': (filtered_total / original_total) if original_total else 0,
                    'pure_pass_rate': pure_pass_rate,
                    'pure_true_count': pure_true,
                    'pure_total_count': pure_total,
                    'true_pass_rate': true_pass_rate,  # corrected rate including catastrophic as 0%
                    'all_true_count': all_true,
                    'all_total_count': all_total
                }

                # accumulate for macro / weighted (filtered) - legacy
                if filtered_total > 0:
                    weighted_true_sum += filtered_true
                    weighted_total_sum += filtered_total
                # accumulate for true macro / weighted (including catastrophic as 0%)
                if all_total > 0:
                    all_weighted_true_sum += all_true
                    all_weighted_total_sum += all_total
                # accumulate for pure-only metrics
                if pure_total > 0:
                    pure_weighted_true_sum += pure_true
                    pure_weighted_total_sum += pure_total

            # macro averages (corrected to include catastrophic as 0%)
            true_pass_rates = [v['true_pass_rate'] for v in option_scores.values() if v['all_total_count'] > 0]
            if true_pass_rates:
                macro_score = float(np.mean(true_pass_rates))
            else:
                macro_score = 0.0
            weighted_macro_score = (all_weighted_true_sum / all_weighted_total_sum * 100) if all_weighted_total_sum > 0 else 0.0

            pure_pass_rates = [v['pure_pass_rate'] for v in option_scores.values() if v['pure_total_count'] > 0]
            if pure_pass_rates:
                non_ocr_macro_score = float(np.mean(pure_pass_rates))
            else:
                non_ocr_macro_score = 0.0
            non_ocr_weighted_macro_score = (pure_weighted_true_sum / pure_weighted_total_sum * 100) if pure_weighted_total_sum > 0 else 0.0

            total_annotations = len(method_data)
            catastrophic_count = len(catastrophic_annotations)
            mixed_count = len(mixed_annotations)
            pure_count = len(pure_annotations)
            catastrophic_rate = catastrophic_count / total_annotations if total_annotations else 0
            mixed_rate = mixed_count / total_annotations if total_annotations else 0
            pure_rate = pure_count / total_annotations if total_annotations else 0

            # composite reliability adjustment
            composite_score = macro_score * (1 - (catastrophic_rate + 0.5 * mixed_rate))
            if composite_score < 0:
                composite_score = 0.0

            method_scores[method] = {
                'macro_score': macro_score,
                'weighted_macro_score': weighted_macro_score,
                'non_ocr_macro_score': non_ocr_macro_score,
                'non_ocr_weighted_macro_score': non_ocr_weighted_macro_score,
                'composite_score': composite_score,
                'option_scores': option_scores,
                'total_annotations': total_annotations,
                'ocr_penalty': {
                    'catastrophic_count': catastrophic_count,
                    'mixed_count': mixed_count,
                    'pure_count': pure_count,
                    'catastrophic_rate': catastrophic_rate,
                    'mixed_rate': mixed_rate,
                    'pure_rate': pure_rate
                }
            }
        return method_scores
    
    def calculate_option_cross_method_scores(self) -> Dict[str, Dict]:
        """Calculate option performance across methods."""
        option_cross_method = {}
        for option in self.all_options:
            method_scores = {}
            for method in self.methods:
                method_data = [ann for ann in self.data 
                              if self._get_method_for_annotation(ann) == method]
                # find annotations where this option was available
                available_annotations = [
                    ann for ann in method_data 
                    if any(opt['id'] == option for opt in ann.get('options', []))
                ]
                if available_annotations:
                    true_count = sum(1 for ann in available_annotations 
                                   if option in ann.get('accept', []))
                    total_count = len(available_annotations)
                    pass_rate = (true_count / total_count * 100) if total_count > 0 else 0
                    method_scores[method] = {
                        'pass_rate': pass_rate,
                        'true_count': true_count,
                        'total_count': total_count
                    }
            if method_scores:
                option_cross_method[option] = method_scores
        return option_cross_method
    
    def calculate_poem_winners(self) -> Dict:
        """Calculate poem-level winners and rankings with catastrophic OCR handling.

        Catastrophic annotations (only ocr_error* accepted) are assigned score=0.
        Mixed annotations retain full (#accepts / #options) score (no penalty scaling).
        """
        poem_results = {}
        method_wins = defaultdict(int)
        method_option_wins = defaultdict(lambda: defaultdict(int))
        method_wisp_wins = defaultdict(lambda: defaultdict(int))
        # group by poem
        poem_data = defaultdict(list)
        for ann in self.data:
            poem_data[ann['filename']].append(ann)
        for poem, annotations in poem_data.items():
            poem_scores = {}
            # calculate score for each method-poem combination
            for ann in annotations:
                method = self._get_method_for_annotation(ann)
                if not method:
                    continue
                options_ids = [opt['id'] for opt in ann.get('options', [])]
                total_options = len(options_ids)
                if total_options == 0:
                    continue
                accepts = ann.get('accept', []) or []
                ocr_accept = any(a.startswith('ocr_error') for a in accepts)
                non_ocr_accepts = [a for a in accepts if not a.startswith('ocr_error')]
                catastrophic = ocr_accept and not non_ocr_accepts
                if catastrophic:
                    score = 0.0
                    true_accepts = 0
                else:
                    true_accepts = len(accepts)  # keep original numerator including OCR accepts
                    score = true_accepts / total_options
                poem_scores[method] = {
                    'score': score,
                    'true_accepts': true_accepts,
                    'total_options': total_options,
                    'accepts': accepts,
                    'options': options_ids,
                    'catastrophic': catastrophic
                }
            # determine winner(s) - methods with maximum score
            if poem_scores:
                max_score = max(poem_scores[method]['score'] for method in poem_scores)
                winners = [method for method, data in poem_scores.items() 
                          if data['score'] == max_score]
                # award wins ...
                for winner in winners:
                    method_wins[winner] += 1
                    # ... by option
                    winner_accepts = poem_scores[winner]['accepts']
                    for accept in winner_accepts:
                        method_option_wins[winner][accept] += 1
                    # ... by wisp label
                    for accept in winner_accepts:
                        for wisp_label, options in self.wisp_label_mapping.items():
                            if accept in options:
                                method_wisp_wins[winner][wisp_label] += 1
                poem_results[poem] = {
                    'scores': poem_scores,
                    'winners': winners,
                    'max_score': max_score
                }
        return {
            'poem_results': poem_results,
            'method_wins': dict(method_wins),
            'method_option_wins': {k: dict(v) for k, v in method_option_wins.items()},
            'method_wisp_wins': {k: dict(v) for k, v in method_wisp_wins.items()}
        }
    
    def calculate_wisp_label_scores(self) -> Dict[str, Dict]:
        """Calculate macro scores by wisp label."""
        wisp_scores = {}
        for wisp_label, options in self.wisp_label_mapping.items():
            method_scores = {}
            for method in self.methods:
                method_data = [ann for ann in self.data 
                              if self._get_method_for_annotation(ann) == method]
                option_pass_rates = []
                option_details = {}
                # calculate pass rate for each option in this wisp label
                for option in options:
                    available_annotations = [
                        ann for ann in method_data 
                        if any(opt['id'] == option for opt in ann.get('options', []))
                    ]
                    if available_annotations:
                        true_count = sum(1 for ann in available_annotations 
                                       if option in ann.get('accept', []))
                        total_count = len(available_annotations)
                        pass_rate = (true_count / total_count * 100) if total_count > 0 else 0
                        option_pass_rates.append(pass_rate)
                        option_details[option] = {
                            'pass_rate': pass_rate,
                            'true_count': true_count,
                            'total_count': total_count
                        }
                # macro-average across options in this wisp label
                if option_pass_rates:
                    wisp_macro_score = np.mean(option_pass_rates)
                    method_scores[method] = {
                        'macro_score': wisp_macro_score,
                        'option_count': len(option_pass_rates),
                        'option_details': option_details
                    }
            if method_scores:
                wisp_scores[wisp_label] = method_scores
        return wisp_scores
    
    def generate_report(self) -> Dict:
        """Generate comprehensive scoring report."""
        print("Calculating method scores...")
        method_scores = self.calculate_method_scores()
        print("Calculating cross-method option scores...")
        option_scores = self.calculate_option_cross_method_scores()
        print("Calculating poem winners...")
        winner_analysis = self.calculate_poem_winners()
        print("Calculating wisp label scores...")
        wisp_scores = self.calculate_wisp_label_scores()
        # create rankings
        method_ranking = sorted(
            method_scores.items(), 
            key=lambda x: x[1]['macro_score'], 
            reverse=True
        )
        # ranking by weighted macro (micro-style) score
        weighted_method_ranking = sorted(
            method_scores.items(),
            key=lambda x: x[1]['weighted_macro_score'],
            reverse=True
        )
        composite_method_ranking = sorted(
            method_scores.items(),
            key=lambda x: x[1]['composite_score'],
            reverse=True
        )
        wins_ranking = sorted(
            winner_analysis['method_wins'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        # create option ranking (by average performance across methods)
        option_ranking = []
        for option, method_scores_dict in option_scores.items():
            avg_score = np.mean([scores['pass_rate'] for scores in method_scores_dict.values()])
            option_ranking.append((option, avg_score, len(method_scores_dict)))
        option_ranking.sort(key=lambda x: x[1], reverse=True)
        # compile final report
        report = {
            'meta': {
                'total_annotations': len(self.data),
                'unique_poems': len(self.all_poems),
                'methods': list(self.methods),
                'total_options': list(self.all_options),
                'wisp_label_mapping': {k: list(v) for k, v in self.wisp_label_mapping.items()},
                'annotator_counts': self.annotator_counts
            },
            'method_scores': method_scores,
            'method_ranking': method_ranking,
            'weighted_method_ranking': weighted_method_ranking,
            'composite_method_ranking': composite_method_ranking,
            'option_cross_method_scores': option_scores,
            'option_ranking': option_ranking,
            'winner_analysis': winner_analysis,
            'wins_ranking': wins_ranking,
            'wisp_label_scores': wisp_scores
        }
        return report
    
    def print_summary(self, report: Dict):
        """Print a summary of key results."""
        print("\n" + "="*60)
        print("WISP-BENCH SCORING SUMMARY")
        print("="*60)
        print("Note: macro & weighted exclude catastrophic OCR rows; composite applies reliability discount cat + 0.5*mixed. Non-OCR metrics use only pure rows.")

        # dataset meta
        print(f"\nDataset: {report['meta']['total_annotations']} annotations, "
              f"{report['meta']['unique_poems']} poems, "
              f"{len(report['meta']['methods'])} methods")

        # method rankings
        print("\nMETHOD RANKING (Macro-averaged across options, olmOCR-Bench style):")
        print("-" * 60)
        for i, (method, data) in enumerate(report['method_ranking'], 1):
            print(f"{i:2d}. {method:20s} {data['macro_score']:6.2f}% "
                  f"({len(data['option_scores'])} options, {data['total_annotations']} annotations)")

        print("\nWEIGHTED METHOD RANKING (Weighted by available annotations per option):")
        print("-" * 60)
        for i, (method, data) in enumerate(report['weighted_method_ranking'], 1):
            print(f"{i:2d}. {method:20s} {data['weighted_macro_score']:6.2f}% "
                  f"(weighted / micro over {len(data['option_scores'])} options)")

        print("\nCOMPOSITE METHOD RANKING (Reliability-adjusted):")
        print("-" * 60)
        for i, (method, data) in enumerate(report['composite_method_ranking'], 1):
            pen = data['ocr_penalty']
            print(f"{i:2d}. {method:20s} {data['composite_score']:6.2f}%  cat {pen['catastrophic_rate']*100:5.1f}%  mix {pen['mixed_rate']*100:5.1f}%  pure {pen['pure_rate']*100:5.1f}%")

        print("\nNON-OCR (PURE) PERFORMANCE (Macro then Weighted):")
        print("-" * 60)
        for method, data in report['method_scores'].items():
            print(f"{method:20s} macro {data['non_ocr_macro_score']:6.2f}%  weighted {data['non_ocr_weighted_macro_score']:6.2f}%  pure_rows {data['ocr_penalty']['pure_count']:3d}")

        # poem winner ranking
        print("\nPOEM WINNER RANKING (Highest true/total ratio per poem):")
        print("-" * 60)
        for i, (method, wins) in enumerate(report['wins_ranking'], 1):
            total_poems = report['meta']['unique_poems']
            win_rate = (wins / total_poems) * 100 if total_poems else 0
            print(f"{i:2d}. {method:20s} {wins:3d} wins ({win_rate:5.1f}%)")

        # option performance
        print("\nOPTION PERFORMANCE RANKING (Average across methods):")
        print("-" * 60)
        for i, (option, avg_score, method_count) in enumerate(report['option_ranking'][:10], 1):
            print(f"{i:2d}. {option:30s} {avg_score:6.2f}% (across {method_count} methods)")

        # WISP label performance
        print("\nWISP LABEL PERFORMANCE:")
        print("-" * 60)
        for wisp_label, method_scores in report['wisp_label_scores'].items():
            print(f"\n{wisp_label} ({len(report['meta']['wisp_label_mapping'][wisp_label])} options):")
            wisp_ranking = sorted(method_scores.items(), key=lambda x: x[1]['macro_score'], reverse=True)
            for method, data in wisp_ranking:
                print(f"  {method:18s} {data['macro_score']:6.2f}% ({data['option_count']} options)")

        # WISP label winner rankings
        print("\nWISP LABEL WINNER RANKINGS:")
        print("-" * 60)
        for wisp_label in report['wisp_label_scores'].keys():
            wisp_wins = defaultdict(int)
            for method, wins_by_wisp in report['winner_analysis']['method_wisp_wins'].items():
                wisp_wins[method] = wins_by_wisp.get(wisp_label, 0)
            wisp_wins_ranking = sorted(wisp_wins.items(), key=lambda x: x[1], reverse=True)
            if wisp_wins_ranking:
                print(f"\n{wisp_label}:")
                for method, wins in wisp_wins_ranking[:3]:  # Top 3
                    if wins > 0:
                        print(f"  {method:18s} {wins:3d} wins")
    
    def export_results(self, report: Dict, output_file: str, format: str = 'json'):
        """Export results to file."""
        output_path = Path(output_file)
        if format.lower() == 'json':
            with open(output_path.with_suffix('.json'), 'w', encoding='utf-8') as f:
                # Convert numpy types to native Python for JSON serialization
                def convert_numpy(obj):
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, np.floating):
                        return float(obj)
                    elif isinstance(obj, np.integer):
                        return int(obj)
                    return obj
                json.dump(report, f, indent=2, ensure_ascii=False, default=convert_numpy)
            print(f"Results exported to {output_path.with_suffix('.json')}")


def main():
    parser = argparse.ArgumentParser(description='WISP-Bench Scoring System')
    parser.add_argument('jsonl_file', help='Path to JSONL annotation file')
    parser.add_argument('--export', type=str, help='Directory to write results (filename same as input JSONL)')
    parser.add_argument('--format', choices=['json'], default='json', help='Export format')
    parser.add_argument('--detailed', action='store_true', help='Show detailed results')
    args = parser.parse_args()
    
    # initialize scorer
    print(f"Loading data from {args.jsonl_file}...")
    scorer = WISPBenchScorer(args.jsonl_file)
    # generate report
    print("Generating report...")
    report = scorer.generate_report()
    # print summary
    scorer.print_summary(report)
    # show detailed results if requested
    if args.detailed:
        print("\n" + "="*60)
        print("DETAILED RESULTS")
        print("="*60)
        print(json.dumps(report, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x))
    # export if requested
    if args.export:
        export_dir = Path(args.export)
        export_dir.mkdir(parents=True, exist_ok=True)
        base_name = Path(args.jsonl_file).stem  # drop .jsonl extension
        output_path = export_dir / base_name
        scorer.export_results(report, str(output_path), args.format)


if __name__ == "__main__":
    main()