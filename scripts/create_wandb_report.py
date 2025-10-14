#!/usr/bin/env python3
"""Create a comprehensive W&B report for OSV5M training runs.

This script creates a structured dashboard with panels organized by metric category
to track holistic model improvements over time.
"""

import argparse
import json
import wandb
import wandb.apis.reports as wr


def create_osv5m_report(entity: str, project: str, title: str = "OSV5M Training Dashboard", 
                        description: str = None, group: str = None):
    """Create a comprehensive W&B report for OSV5M training.
    
    Args:
        entity: W&B entity (username or team)
        project: W&B project name
        title: Report title
        description: Report description
        group: Optional group filter for runs
    
    Returns:
        Report URL
    """
    
    # Initialize W&B API
    api = wandb.Api()
    
    # Build run filter
    filters = {"$and": []}
    if group:
        filters["$and"].append({"group": group})
    
    # Create report
    report = wr.Report(
        entity=entity,
        project=project,
        title=title,
        description=description or "Comprehensive dashboard for tracking OSV5M model improvements"
    )
    
    # Panel specifications organized by category
    panels = []
    
    # ============= HOLISTIC PERFORMANCE OVERVIEW =============
    # Main accuracy metrics showing overall improvement
    panels.append(
        wr.LinePlot(
            title="📊 Overall Task Performance",
            x="global_step",
            y=["env/stage_acc", "env/is_correct", "env/any_correct"],
            title_x="Training Step",
            title_y="Accuracy",
            smoothing_factor=0.8,
            max_runs_to_show=10,
            plot_type="line",
            font_size="auto",
            legend_position="east"
        )
    )
    
    # Hierarchical accuracy breakdown
    panels.append(
        wr.LinePlot(
            title="🌍 Hierarchical Location Accuracy",
            x="global_step",
            y=["env/country_correct", "env/region_correct", "env/city_correct"],
            title_x="Training Step",
            title_y="Accuracy",
            smoothing_factor=0.8,
            max_runs_to_show=10,
            plot_type="line",
            font_size="auto",
            legend_position="east"
        )
    )
    
    # ============= COORDINATE PREDICTION QUALITY =============
    # Distance metrics showing coordinate prediction improvement
    panels.append(
        wr.LinePlot(
            title="📍 Coordinate Prediction - Distance Error",
            x="global_step",
            y=["env/distance_km"],
            title_x="Training Step",
            title_y="Distance Error (km)",
            smoothing_factor=0.8,
            max_runs_to_show=10,
            plot_type="line",
            font_size="auto",
            legend_position="east"
        )
    )
    
    # Coordinate quality indicators
    panels.append(
        wr.LinePlot(
            title="✅ Coordinate Quality Indicators",
            x="global_step",
            y=["env/has_coords", "env/coords_in_range", "env/coords_within_tolerance"],
            title_x="Training Step",
            title_y="Proportion",
            smoothing_factor=0.8,
            max_runs_to_show=10,
            plot_type="line",
            font_size="auto",
            legend_position="east"
        )
    )

    # Distance threshold progression (NEW HOLISTIC METRICS)
    panels.append(
        wr.LinePlot(
            title="🎯 Distance Threshold Pass Rates",
            x="global_step",
            y=["env/within_10km", "env/within_25km", "env/within_50km", "env/within_100km",
               "env/within_250km", "env/within_500km", "env/within_1000km"],
            title_x="Training Step",
            title_y="Pass Rate (0-1)",
            smoothing_factor=0.8,
            max_runs_to_show=10,
            plot_type="line",
            font_size="auto",
            legend_position="east"
        )
    )

    # Distance distribution percentiles
    panels.append(
        wr.LinePlot(
            title="📊 Distance Error Distribution (Percentiles)",
            x="global_step",
            y=["env/distance_p25", "env/distance_p50", "env/distance_p75", "env/distance_p90"],
            title_x="Training Step",
            title_y="Distance (km)",
            smoothing_factor=0.8,
            max_runs_to_show=10,
            plot_type="line",
            font_size="auto",
            legend_position="east"
        )
    )

    # ============= HOLISTIC IMPROVEMENT SCORES =============
    # Overall improvement score - single metric combining all aspects
    panels.append(
        wr.LinePlot(
            title="🌟 Overall Improvement Score",
            x="global_step",
            y=["env/overall_improvement_score"],
            title_x="Training Step",
            title_y="Score (0-1)",
            smoothing_factor=0.8,
            max_runs_to_show=10,
            plot_type="line",
            font_size="auto",
            legend_position="east"
        )
    )

    # Composite health scores
    panels.append(
        wr.LinePlot(
            title="💯 Composite Health Scores",
            x="global_step",
            y=["env/format_health_score", "env/hierarchical_score", "env/coord_quality_score"],
            title_x="Training Step",
            title_y="Score (0-1)",
            smoothing_factor=0.8,
            max_runs_to_show=10,
            plot_type="line",
            font_size="auto",
            legend_position="east"
        )
    )

    # Success metrics
    panels.append(
        wr.LinePlot(
            title="✨ Success Rate Metrics",
            x="global_step",
            y=["env/success_rate", "env/high_confidence_rate"],
            title_x="Training Step",
            title_y="Rate (0-1)",
            smoothing_factor=0.8,
            max_runs_to_show=10,
            plot_type="line",
            font_size="auto",
            legend_position="east"
        )
    )

    # ============= REWARD AND ADVANTAGE STATISTICS =============
    # Raw reward statistics before normalization
    panels.append(
        wr.LinePlot(
            title="💰 Reward Statistics (Pre-Normalization)",
            x="global_step",
            y=["rewards/mean", "rewards/median", "rewards/std"],
            title_x="Training Step",
            title_y="Value",
            smoothing_factor=0.8,
            max_runs_to_show=10,
            plot_type="line",
            font_size="auto",
            legend_position="east"
        )
    )

    # Reward range (min/max)
    panels.append(
        wr.LinePlot(
            title="📈 Reward Range (Min/Max)",
            x="global_step",
            y=["rewards/min", "rewards/max"],
            title_x="Training Step",
            title_y="Reward",
            smoothing_factor=0.8,
            max_runs_to_show=10,
            plot_type="line",
            font_size="auto",
            legend_position="east"
        )
    )

    # Advantage statistics after normalization
    panels.append(
        wr.LinePlot(
            title="⚡ Advantage Statistics (Post-Normalization)",
            x="global_step",
            y=["advantages/mean", "advantages/std", "advantages/min", "advantages/max"],
            title_x="Training Step",
            title_y="Value",
            smoothing_factor=0.8,
            max_runs_to_show=10,
            plot_type="line",
            font_size="auto",
            legend_position="east"
        )
    )

    # ============= FORMAT AND PARSING SUCCESS =============
    # Format correctness tracking
    panels.append(
        wr.LinePlot(
            title="📝 Response Format Quality",
            x="global_step",
            y=["env/format_ok", "env/parsed_field_count"],
            title_x="Training Step",
            title_y="Format Score",
            smoothing_factor=0.8,
            max_runs_to_show=10,
            plot_type="line",
            font_size="auto",
            legend_position="east"
        )
    )
    
    # Field presence tracking
    panels.append(
        wr.LinePlot(
            title="🔍 Structured Field Detection",
            x="global_step",
            y=["env/has_country", "env/has_region", "env/has_city", "env/country_valid"],
            title_x="Training Step",
            title_y="Field Presence",
            smoothing_factor=0.8,
            max_runs_to_show=10,
            plot_type="line",
            font_size="auto",
            legend_position="east"
        )
    )
    
    # ============= REWARD AND CURRICULUM =============
    # Reward decomposition
    panels.append(
        wr.LinePlot(
            title="🎯 Reward Components",
            x="global_step",
            y=["env/final_reward", "env/hier_reward", "env/geo_reward"],
            title_x="Training Step",
            title_y="Reward",
            smoothing_factor=0.8,
            max_runs_to_show=10,
            plot_type="line",
            font_size="auto",
            legend_position="east"
        )
    )
    
    # Curriculum progression
    panels.append(
        wr.LinePlot(
            title="📈 Curriculum Stage Progression",
            x="global_step",
            y=["env/stage_idx"],
            title_x="Training Step",
            title_y="Stage Index",
            smoothing_factor=0.0,  # No smoothing for discrete stages
            max_runs_to_show=10,
            plot_type="line",
            font_size="auto",
            legend_position="east"
        )
    )
    
    # Curriculum weights
    panels.append(
        wr.LinePlot(
            title="⚖️ Curriculum Target Weights",
            x="global_step",
            y=["env/stage_country_weight", "env/stage_region_weight", "env/stage_city_weight"],
            title_x="Training Step",
            title_y="Weight",
            smoothing_factor=0.0,
            max_runs_to_show=10,
            plot_type="line",
            font_size="auto",
            legend_position="east"
        )
    )
    
    # ============= RESPONSE LENGTH ANALYSIS =============
    panels.append(
        wr.LinePlot(
            title="📏 Response Length Trends",
            x="global_step",
            y=["env/response_len_tokens", "env/response_len_chars"],
            title_x="Training Step",
            title_y="Length",
            smoothing_factor=0.8,
            max_runs_to_show=10,
            plot_type="line",
            font_size="auto",
            legend_position="east"
        )
    )
    
    # ============= TRAINING HEALTH METRICS =============
    # PPO health indicators
    panels.append(
        wr.LinePlot(
            title="🏥 PPO Training Health",
            x="global_step",
            y=["loss", "entropy", "ppo/approx_kl", "ppo/ratio_clipped_frac"],
            title_x="Training Step",
            title_y="Value",
            smoothing_factor=0.8,
            max_runs_to_show=10,
            plot_type="line",
            font_size="auto",
            legend_position="east"
        )
    )
    
    # Training efficiency
    panels.append(
        wr.LinePlot(
            title="⚡ Training Efficiency",
            x="global_step",
            y=["times/time_per_rollout", "times/time_per_inference_iteration"],
            title_x="Training Step",
            title_y="Time (seconds)",
            smoothing_factor=0.8,
            max_runs_to_show=10,
            plot_type="line",
            font_size="auto",
            legend_position="east"
        )
    )
    
    # Learning rate tracking
    panels.append(
        wr.LinePlot(
            title="📉 Learning Rate",
            x="global_step",
            y=["training/learning_rate"],
            title_x="Training Step",
            title_y="Learning Rate",
            smoothing_factor=0.0,
            max_runs_to_show=10,
            plot_type="line",
            font_size="auto",
            legend_position="east"
        )
    )
    
    # ============= COMPARATIVE ANALYSIS =============
    # Bar chart for final performance comparison across runs
    panels.append(
        wr.BarPlot(
            title="🏆 Final Performance Comparison",
            metrics=["env/stage_acc", "env/country_correct", "env/region_correct", "env/city_correct"],
            orientation="v",
            title_x="Metric",
            title_y="Final Value",
            max_runs_to_show=10,
            font_size="auto"
        )
    )
    
    # Scatter plot for reward vs accuracy correlation
    panels.append(
        wr.ScatterPlot(
            title="🔄 Reward vs Accuracy Correlation",
            x="env/final_reward",
            y="env/stage_acc",
            running_ymin=True,
            running_ymax=True
        )
    )
    
    # ============= SUMMARY STATISTICS =============
    # Run comparison table
    panels.append(
        wr.RunComparer(
            diff_only="split",
            layout={"w": 24, "h": 8}
        )
    )
    
    # Parallel coordinates for multi-metric comparison
    panels.append(
        wr.ParallelCoordinatesPlot(
            columns=[
                {"metric": "env/overall_improvement_score"},
                {"metric": "env/stage_acc"},
                {"metric": "env/country_correct"},
                {"metric": "env/region_correct"},
                {"metric": "env/city_correct"},
                {"metric": "env/format_ok"},
                {"metric": "env/coords_within_tolerance"},
                {"metric": "env/format_health_score"},
                {"metric": "env/hierarchical_score"},
                {"metric": "env/coord_quality_score"},
                {"metric": "env/within_100km"},
                {"metric": "env/distance_km"},
                {"metric": "env/final_reward"}
            ],
            title="🌐 Multi-Metric Performance Overview",
            layout={"w": 24, "h": 8}
        )
    )
    
    # Add all panels to report
    report.blocks = [wr.PanelGrid(panels=panels, runsets=[wr.Runset(entity=entity, project=project, filters=json.dumps(filters) if filters["$and"] else None)])]
    
    # Save report
    report.save()
    
    print(f"✅ Report created successfully!")
    print(f"📊 View your report at: {report.url}")
    
    return report.url


def add_distance_thresholds(entity: str, project: str):
    """Log additional distance threshold metrics to existing runs.
    
    This function adds binary metrics for common distance thresholds
    to make it easier to track pass/fail rates at different tolerances.
    """
    
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}")
    
    print("Adding distance threshold metrics to runs...")
    
    for run in runs:
        # Check if run has distance data
        if "env/distance_km" not in run.summary:
            continue
            
        # Get distance history
        history = run.scan_history(keys=["env/distance_km", "global_step"])
        
        # Calculate threshold metrics
        for row in history:
            if "env/distance_km" in row and row["env/distance_km"] is not None:
                distance = row["env/distance_km"]
                step = row.get("global_step", 0)
                
                # Log threshold metrics
                with wandb.init(id=run.id, resume="allow", project=project, entity=entity) as resumed_run:
                    resumed_run.log({
                        "env/within_10km": int(distance <= 10),
                        "env/within_25km": int(distance <= 25),
                        "env/within_50km": int(distance <= 50),
                        "env/within_100km": int(distance <= 100),
                        "env/within_250km": int(distance <= 250),
                        "env/within_500km": int(distance <= 500),
                        "env/within_1000km": int(distance <= 1000),
                        "global_step": step
                    }, commit=False)
    
    print("✅ Distance thresholds added!")


def main():
    parser = argparse.ArgumentParser(description="Create W&B report for OSV5M training")
    parser.add_argument("--entity", type=str, required=True, help="W&B entity (username or team)")
    parser.add_argument("--project", type=str, required=True, help="W&B project name")
    parser.add_argument("--title", type=str, default="OSV5M Training Dashboard", help="Report title")
    parser.add_argument("--description", type=str, help="Report description")
    parser.add_argument("--group", type=str, help="Filter runs by group")
    parser.add_argument("--add-thresholds", action="store_true", help="Add distance threshold metrics to runs")
    
    args = parser.parse_args()
    
    # Optionally add distance thresholds first
    if args.add_thresholds:
        add_distance_thresholds(args.entity, args.project)
    
    # Create the report
    url = create_osv5m_report(
        entity=args.entity,
        project=args.project,
        title=args.title,
        description=args.description,
        group=args.group
    )
    
    print(f"\n🎉 Dashboard created successfully!")
    print(f"📊 Open your dashboard: {url}")
    print("\n💡 Tips for using the dashboard:")
    print("  - Use the run selector to compare different training configurations")
    print("  - Click on legend items to show/hide specific metrics")
    print("  - Adjust smoothing to see trends vs raw values")
    print("  - Use the parallel coordinates plot to identify high-performing runs")
    print("\n📈 Key holistic metrics to watch for improvement:")
    print("  - env/overall_improvement_score: Single metric combining all aspects (0-1)")
    print("  - env/format_health_score: Aggregate of all format indicators (0-1)")
    print("  - env/hierarchical_score: Weighted country→region→city progression (0-1)")
    print("  - env/coord_quality_score: Coordinate prediction quality (0-1)")
    print("  - env/within_100km: Pass rate at 100km threshold (watch this climb!)")
    print("  - env/distance_km: Average prediction error in km (lower is better)")
    print("\n🎯 What good training looks like:")
    print("  - Distance thresholds improving in sequence: 1000km → 500km → 100km → 50km")
    print("  - Format health score steadily increasing toward 1.0")
    print("  - Overall improvement score trending upward consistently")
    print("  - Reward variance decreasing as model learns the task")


if __name__ == "__main__":
    main()