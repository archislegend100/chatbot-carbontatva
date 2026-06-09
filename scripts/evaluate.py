import json
import asyncio
import time
import argparse
from rich.console import Console
from app.routing.query_router import query_router

console = Console()

def evaluate_smoke():
    with open("backend/app/evaluation/test_queries.json", "r") as f:
        queries = json.load(f)
        
    console.print(f"[bold green]Starting Smoke Evaluation ({len(queries)} queries)[/bold green]")
    
    for q in queries:
        query_text = q["query"]
        expected = q["category"]
        
        t0 = time.time()
        profile = query_router.route_query(query_text)
        dt = (time.time() - t0) * 1000
        
        console.print(f"\n[cyan]Query:[/cyan] {query_text}")
        console.print(f"[yellow]Expected Category:[/yellow] {expected}")
        console.print(f"[magenta]Routed Intent:[/magenta] {profile.intent}")
        console.print(f"[magenta]Domain:[/magenta] {profile.utility_domain}")
        console.print(f"Routing Latency: {dt:.2f}ms")
        
        if profile.intent == expected or (expected == "qa" and profile.intent in ["explainer", "qa"]):
            console.print("[green]Routing matched expected category.[/green]")
        else:
            console.print("[red]Routing mismatch![/red]")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Run smoke evaluation tests.")
    args = parser.parse_args()
    
    if args.smoke:
        evaluate_smoke()
    else:
        console.print("Please provide --smoke to run evaluation.")
