"""
Workflow CLI - Simple Command Line Interface for Ichimoku Trading System

This module provides a user-friendly command-line interface to interact with
streamlined data collection utilities in this folder.

Features:
- Interactive menu system
- Input validation
- Progress indicators
- Result display
"""

import sys
import time
from pathlib import Path
from typing import List, Any

# Ensure backend is on the path so "streamline_workflow" package imports work when running directly
sys.path.append(str(Path(__file__).parent.parent))

from data_fetching.collect_historical_data import collect_all_historical_data_for_all_pairs
from strategy.compute_ichimoku_to_sql import compute_for_symbol
from backtesting.ichimoku_backtester import IchimokuBacktester, StrategyBacktestRunner
from pathlib import Path
import json

# ANSI color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class WorkflowCLI:
    """
    Command-line interface for the streamlined workflow (data collection and indicator computation).
    """
    
    def __init__(self):
        """Initialize the CLI interface."""
        self.session_active = False
        
    def print_header(self):
        """Print the application header."""
        print("\n" + "="*70)
        print(f"{Colors.BOLD}{Colors.HEADER}🌊 ICHIMOKU CLOUD TRADING SYSTEM 🌊{Colors.ENDC}")
        print("="*70)
        print(f"{Colors.CYAN}Streamlined Workflow: Data Collection{Colors.ENDC}")
        print("="*70 + "\n")
    
    def print_menu(self):
        """Display the main menu."""
        print(f"\n{Colors.BOLD}=== MAIN MENU ==={Colors.ENDC}")
        print(f"{Colors.GREEN}1.{Colors.ENDC} Collect historical data")
        print(f"{Colors.GREEN}2.{Colors.ENDC} Compute Ichimoku")
        print(f"{Colors.GREEN}3.{Colors.ENDC} Backtest Ichimoku Strategy")
        print(f"{Colors.GREEN}0.{Colors.ENDC} Exit")
        print("-"*30)
    
    def get_user_choice(self, prompt: str, valid_choices: List[str]) -> str:
        """Get validated user input."""
        while True:
            choice = input(f"{Colors.CYAN}{prompt}{Colors.ENDC} ").strip()
            if choice in valid_choices:
                return choice
            print(f"{Colors.WARNING}Invalid choice. Please try again.{Colors.ENDC}")
    
    def get_user_input(self, prompt: str, default: Any = None, 
                      input_type: type = str) -> Any:
        """Get user input with type conversion and default values."""
        if default is not None:
            prompt = f"{prompt} [{default}]"
        
        while True:
            user_input = input(f"{Colors.CYAN}{prompt}: {Colors.ENDC}").strip()
            
            # Use default if no input
            if not user_input and default is not None:
                return default
            
            # Convert to appropriate type
            try:
                if input_type == bool:
                    return user_input.lower() in ['yes', 'y', 'true', '1']
                else:
                    return input_type(user_input)
            except ValueError:
                print(f"{Colors.WARNING}Invalid input. Expected {input_type.__name__}.{Colors.ENDC}")
    
    def display_progress(self, message: str):
        """Display a progress indicator."""
        print(f"\n{Colors.BLUE}⏳ {message}...{Colors.ENDC}", end='', flush=True)
        
        # Simple animated dots
        for _ in range(3):
            time.sleep(0.5)
            print(".", end='', flush=True)
        print()
    
    def run_compute_ichimoku(self):
        """Submenu to compute Ichimoku and save to SQL for selected symbols."""
        while True:
            print(f"\n{Colors.BOLD}=== Compute Ichimoku ==={Colors.ENDC}")
            print(f"{Colors.GREEN}1.{Colors.ENDC} BTC/USDT")
            print(f"{Colors.GREEN}2.{Colors.ENDC} ETH/USDT")
            print(f"{Colors.GREEN}3.{Colors.ENDC} SOL/USDT")
            print(f"{Colors.GREEN}4.{Colors.ENDC} All (BTC, ETH, SOL)")
            print(f"{Colors.GREEN}0.{Colors.ENDC} Back")

            sub_choice = self.get_user_choice("Select option: ", ['0', '1', '2', '3', '4'])
            if sub_choice == '0':
                break

            # Map sub-choice to symbols
            if sub_choice == '1':
                symbols = ['BTC']
            elif sub_choice == '2':
                symbols = ['ETH']
            elif sub_choice == '3':
                symbols = ['SOL']
            else:
                symbols = ['BTC', 'ETH', 'SOL']

            # Execute calculations
            for sym in symbols:
                try:
                    self.display_progress(f"Computing Ichimoku for {sym}/USDT")
                    stats = compute_for_symbol(sym)
                    print(f"{Colors.GREEN}✓ Completed {sym}/USDT{Colors.ENDC}")
                    # Print brief stats
                    for tf, tf_stats in stats.items():
                        print(f"  {tf}: inserted={tf_stats.get('inserted', 0)}, errors={tf_stats.get('errors', 0)}")
                except Exception as e:
                    print(f"{Colors.FAIL}Error computing for {sym}: {e}{Colors.ENDC}")

    def run_backtest_menu(self):
        """Interactive backtest runner with automatic reporting."""
        # Load available strategies from known locations (config/ or strategy/config/)
        candidates = [
            Path(__file__).resolve().parent / 'config' / 'strategies.json',
            Path(__file__).resolve().parent / 'config' / 'strategies.yaml',
            Path(__file__).resolve().parent / 'strategy' / 'config' / 'strategies.json',
            Path(__file__).resolve().parent / 'strategy' / 'config' / 'strategies.yaml'
        ]
        strategies = {}
        strategies_file = None
        for p in candidates:
            if p.exists():
                try:
                    with open(p, 'r') as f:
                        if p.suffix == '.json':
                            data = json.load(f)
                        else:
                            import yaml
                            data = yaml.safe_load(f)
                        strategies = data.get('strategies', {})
                        strategies_file = p
                        if strategies:
                            break
                except Exception:
                    continue
        strategy_keys = list(strategies.keys())
        if not strategy_keys:
            print(f"{Colors.FAIL}No strategies found. Expected at:{Colors.ENDC}")
            for p in candidates:
                print(f"  - {p}")
            return

        print(f"\n{Colors.BOLD}=== Backtest Ichimoku Strategy ==={Colors.ENDC}")
        # Select strategy
        print("Available strategies:")
        for idx, key in enumerate(strategy_keys, start=1):
            name = strategies[key].get('name', key)
            print(f"  {idx}. {key} - {name}")
        s_choice = self.get_user_input("Select strategy number", default=1, input_type=int)
        if s_choice < 1 or s_choice > len(strategy_keys):
            print(f"{Colors.WARNING}Invalid selection{Colors.ENDC}")
            return
        strategy_key = strategy_keys[s_choice - 1]

        # Select symbol
        print("\nSelect symbol:")
        print(f"{Colors.GREEN}1.{Colors.ENDC} BTC/USDT")
        print(f"{Colors.GREEN}2.{Colors.ENDC} ETH/USDT")
        print(f"{Colors.GREEN}3.{Colors.ENDC} SOL/USDT")
        sym_choice = self.get_user_choice("Option", ['1','2','3'])
        symbol_map = {'1': 'BTC', '2': 'ETH', '3': 'SOL'}
        symbol_short = symbol_map[sym_choice]

        # Select timeframe
        print("\nSelect timeframe:")
        print(f"{Colors.GREEN}1.{Colors.ENDC} 4h")
        print(f"{Colors.GREEN}2.{Colors.ENDC} 1d")
        tf_choice = self.get_user_choice("Option", ['1','2'])
        timeframe = {'1':'4h','2':'1d'}[tf_choice]

        # Optional start date
        start_date = self.get_user_input("Start date (YYYY-MM-DD) or empty for all", default="", input_type=str)
        start_date = start_date.strip().strip('()') if start_date else None

        # Ask whether to recompute Ichimoku with strategy parameters
        recompute_ans = self.get_user_input("Recompute Ichimoku with strategy params? [Y/n]", default="Y", input_type=str)
        force_recompute = str(recompute_ans).strip().lower() in ['y', 'yes', '']

        # Run backtest
        print(f"\n{Colors.BLUE}Running backtest for {symbol_short}/USDT {timeframe} using {strategy_key}{Colors.ENDC}")
        backtester = IchimokuBacktester()
        runner = StrategyBacktestRunner(backtester)
        reports_dir = str(Path(__file__).resolve().parent / 'reports')
        try:
                outcome = runner.run_from_json(
                    strategy_key=strategy_key,
                    symbol_short=symbol_short,
                    timeframe=timeframe,
                    start=start_date,
                    end=None,
                    initial_capital=10000.0,
                    report_formats='pdf',
                    output_dir=reports_dir,
                    force_recompute_ichimoku=force_recompute
                )
                print(f"{Colors.GREEN}Backtest completed. Reports:{Colors.ENDC}")
                for k, v in outcome['reports'].items():
                    if isinstance(v, list):
                        for p in v:
                            print(f"  {k}: {p}")
                    else:
                        print(f"  {k}: {v}")
                # Offer optional LLM optimization step AFTER report generation
                do_llm = self.get_user_input("Generate LLM Optimization Report (PDF-only)? [y/N]", default="N", input_type=str)
                if str(do_llm).strip().lower() in ['y','yes']:
                    print(f"\n{Colors.BOLD}=== LLM Optimization ==={Colors.ENDC}")
                    # Ask for analysis timeframe (optional)
                    use_same = self.get_user_input("Use the same backtest date range? [Y/n]", default="Y", input_type=str)
                    if str(use_same).strip().lower() in ['y','yes','']:
                        analysis_start = start_date
                        analysis_end = None
                    else:
                        analysis_start = self.get_user_input("Analysis start date (YYYY-MM-DD) or empty", default="", input_type=str)
                        analysis_start = analysis_start.strip() or None
                        analysis_end = self.get_user_input("Analysis end date (YYYY-MM-DD) or empty", default="", input_type=str)
                        analysis_end = analysis_end.strip() or None
                    # Variant
                    print("Select prompt variant:")
                    print(f"{Colors.GREEN}1.{Colors.ENDC} Analyst (settings optimization)")
                    print(f"{Colors.GREEN}2.{Colors.ENDC} Risk-focused (drawdown aware)")
                    v_choice = self.get_user_choice("Option", ['1','2'])
                    variant = 'analyst' if v_choice == '1' else 'risk'
                    # Provider override (optional)
                    provider_in = self.get_user_input("Provider override (leave blank for default) [openai|gemini]", default="", input_type=str)
                    provider = provider_in.strip().lower() or None
                    # Model override removed; use configured defaults
                    model_override = None

                    llm_pdf = runner.generate_llm_optimization_report(
                        result=outcome['result'],
                        data_df=outcome['data_df'],
                        trades_df=outcome['trades_df'],
                        equity_df=outcome['equity_df'],
                        strategy_config=outcome['strategy_config'],
                        output_dir=reports_dir,
                        symbol_short=symbol_short,
                        timeframe=timeframe,
                        analysis_start=analysis_start,
                        analysis_end=analysis_end,
                        llm_provider=provider,
                        llm_model_override=model_override,
                        prompt_variant=variant
                    )
                    if llm_pdf:
                        print(f"{Colors.GREEN}LLM PDF:{Colors.ENDC} {llm_pdf}")
                    else:
                        print(f"{Colors.WARNING}LLM Optimization report was not generated.{Colors.ENDC}")
        except Exception as e:
            print(f"{Colors.FAIL}Backtest failed: {e}{Colors.ENDC}")

    def run(self):
        """Run the main CLI loop."""
        self.print_header()
        
        while True:
            self.print_menu()
            
            choice = self.get_user_choice("Select option: ", ['0', '1', '2', '3'])
            
            if choice == '0':
                print(f"\n{Colors.GREEN}Thank you for using Ichimoku Cloud Trading System!{Colors.ENDC}")
                print(f"{Colors.CYAN}Good luck with your trading! 📈{Colors.ENDC}\n")
                break
            elif choice == '1':
                self.display_progress("Collecting historical data")
                collect_all_historical_data_for_all_pairs()
            elif choice == '2':
                self.run_compute_ichimoku()
            elif choice == '3':
                self.run_backtest_menu()
            
            # Pause before returning to menu
            if choice != '0':
                input(f"\n{Colors.CYAN}Press Enter to continue...{Colors.ENDC}")


def main():
    """Main entry point for the CLI."""
    try:
        cli = WorkflowCLI()
        cli.run()
    except KeyboardInterrupt:
        print(f"\n\n{Colors.WARNING}Interrupted by user.{Colors.ENDC}")
        print(f"{Colors.CYAN}Goodbye! 👋{Colors.ENDC}\n")
    except Exception as e:
        print(f"\n{Colors.FAIL}An unexpected error occurred: {e}{Colors.ENDC}")
        print(f"{Colors.WARNING}Please check the logs for details.{Colors.ENDC}")


if __name__ == "__main__":
    main()
