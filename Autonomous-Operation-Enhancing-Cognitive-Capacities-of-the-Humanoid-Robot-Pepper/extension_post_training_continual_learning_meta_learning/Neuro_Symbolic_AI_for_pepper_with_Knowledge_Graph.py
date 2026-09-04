"""
Neuro-Symbolic Pepper Architecture — Complete Implementation

Flow:
User -> post-trained Gemini -> candidate orchestration
     -> perception -> dynamic KG instantiation
     -> symbolic/geometric/kinematic validation
     -> Gemini replanning when necessary
     -> validated orchestration -> Pepper
     -> KG state update

The KG schema is initialized before operation; dynamic scene facts are
populated during operation from perception and computational validators

Gemini/YOLO/IK/ROS/NAOqi calls are represented as integration interfaces.
The implementation provides the neural-symbolic reasoning, grounding, validation, and replanning architecture.
"""

from __future__ import annotations

import json
import math
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Set


# ==========================================================================
# 1. CONFIGURATION
# ==========================================================================

class ExecutionMode(str, Enum):
    SIMULATION = "simulation"
    REAL_ROBOT = "real_robot"


EXECUTION_MODE = ExecutionMode.SIMULATION
GEMINI_MODEL = "gemini-1.5-flash"
MAX_REPLANNING_ROUNDS = 3

# Geometry thresholds used by the symbolic grounding layer.
CONTAINMENT_THRESHOLD = 0.80
COLLISION_CLEARANCE = 0.05


# ============================================================================
# 2. GEOMETRY / PERCEPTION TYPES
# ============================================================================

@dataclass
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)


@dataclass
class Pose3D:
    x: float
    y: float
    z: float
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0


@dataclass
class Detection:
    detection_id: str
    label: str
    confidence: float
    bbox: BoundingBox
    pose_3d: Optional[Pose3D] = None
    source_model: str = "YOLO"


# ============================================================================
# 3. KG DATA MODEL
# ============================================================================

@dataclass
class Entity:
    entity_id: str
    entity_type: str
    label: str
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Relation:
    subject: str
    predicate: str
    object: str
    confidence: float = 1.0
    source: str = "symbolic"


@dataclass
class ConstraintResult:
    name: str
    passed: bool
    explanation: str
    evidence: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationReport:
    valid: bool
    checks: List[ConstraintResult]
    missing_information: List[str] = field(default_factory=list)
    required_replanning: bool = False

    def summary(self) -> str:
        status = "VALID" if self.valid else "INVALID"
        lines = [f"Validation status: {status}"]
        for c in self.checks:
            lines.append(
                f"[{'PASS' if c.passed else 'FAIL'}] "
                f"{c.name}: {c.explanation}"
            )
        if self.missing_information:
            lines.append(
                "Missing information: " +
                ", ".join(self.missing_information)
            )
        return "\n".join(lines)


# ============================================================================
# 4. KG SCHEMA — DEFINED BEFORE ROBOT OPERATION
# ============================================================================

class KnowledgeGraphSchema:
    """
    Defines the symbolic vocabulary. It does not represent the current scene.

    Dynamic object/entity instances are inserted only after observation.
    """

    def __init__(self):
        self.entity_types: Set[str] = {
            "environment", "room", "zone", "surface", "door",
            "object", "container", "human", "robot", "action",
            "model", "sensor", "location", "plan", "trajectory",
            "joint_configuration",
        }

        self.relations: Set[str] = {
            "contains", "inside", "on", "under", "near", "far_from",
            "left_of", "right_of", "in_front_of", "behind",
            "adjacent_to", "connected_to", "located_in", "detected_by",
            "visible_from", "reachable_by", "graspable_by", "requires",
            "requires_model", "precedes", "followed_by", "performed_by",
            "target_of", "owned_by", "held_by", "placed_on",
            "delivered_to", "blocked_by", "collision_with",
            "feasible_for", "infeasible_for", "supports", "has_state",
        }

        self.properties: Set[str] = {
            "position_2d", "position_3d", "orientation", "dimensions",
            "bbox", "confidence", "color", "shape", "weight",
            "movable", "fragile", "graspable", "reachable", "visible",
            "occupied", "joint_limits", "capabilities", "execution_state",
        }

        self.actions: Set[str] = {
            "detect", "locate", "navigate", "approach", "grasp", "pick",
            "place", "release", "give", "deliver", "speak", "inspect",
            "follow", "open", "close",
        }

        self.action_preconditions = {
            "detect": ["target_visible_or_searchable"],
            "navigate": ["destination_known", "navigation_path_exists"],
            "approach": ["target_position_known", "target_reachable"],
            "grasp": [
                "target_position_known",
                "target_reachable",
                "ik_solution_exists",
                "joint_limits_valid",
                "collision_free",
                "grasp_geometry_feasible",
            ],
            "pick": ["grasp_feasible"],
            "place": [
                "holding_target",
                "destination_known",
                "destination_reachable",
            ],
            "give": [
                "holding_target",
                "human_target_known",
                "human_reachable",
            ],
            "deliver": [
                "holding_target",
                "human_target_known",
                "human_reachable",
            ],
        }

        self.rules = [
            "detect_before_manipulation",
            "requested_object_must_exist",
            "spatial_relation_must_be_geometrically_grounded",
            "coordinates_required_before_ik",
            "ik_solution_required_for_manipulation",
            "joint_limits_must_be_satisfied",
            "trajectory_must_be_collision_free",
            "navigation_path_must_exist",
            "manipulation_requires_reachability",
            "grasp_requires_geometry_and_kinematics",
            "delivery_requires_holding",
            "give_requires_human",
            "execution_requires_validated_plan",
        ]

        self.robot_capabilities = {
            "Pepper": {
                "navigation": True,
                "visual_perception": True,
                "speech_recognition": True,
                "speech_synthesis": True,
                "arm_manipulation": True,
                "grasping": True,
                "face_recognition": True,
                "object_detection": True,
            }
        }


# ============================================================================
# 5. KNOWLEDGE GRAPH — EMPTY DYNAMIC WORLD STATE AT STARTUP
# ============================================================================

class KnowledgeGraph:
    def __init__(self, schema: KnowledgeGraphSchema):
        self.schema = schema
        self.entities: Dict[str, Entity] = {}
        self.relations: List[Relation] = []
        self.fact_history: List[Dict[str, Any]] = []
        self._initialize_static_environment()

    def _initialize_static_environment(self):
        self.add_entity(Entity(
            "environment_1", "environment", "Pepper Environment"
        ))

        # Static environment topology can be known before operation.
        for room_id, label in {
            "room_kitchen": "Kitchen",
            "room_living": "Living Room",
            "room_bedroom": "Bedroom",
            "room_hallway": "Hallway",
        }.items():
            self.add_entity(Entity(room_id, "room", label))

        self.add_relation("room_kitchen", "connected_to", "room_hallway")
        self.add_relation("room_living", "connected_to", "room_hallway")
        self.add_relation("room_bedroom", "connected_to", "room_hallway")

        self.add_entity(Entity(
            "pepper",
            "robot",
            "Pepper",
            {"capabilities": self.schema.robot_capabilities["Pepper"]}
        ))

    def add_entity(self, entity: Entity):
        if entity.entity_type not in self.schema.entity_types:
            raise ValueError(f"Unknown entity type: {entity.entity_type}")
        self.entities[entity.entity_id] = entity

    def update_entity_properties(self, entity_id: str, properties: Dict[str, Any]):
        self.entities[entity_id].properties.update(properties)

    def add_relation(
        self,
        subject: str,
        predicate: str,
        object_: str,
        confidence: float = 1.0,
        source: str = "symbolic",
    ):
        if predicate not in self.schema.relations:
            raise ValueError(f"Unknown relation: {predicate}")
        r = Relation(subject, predicate, object_, confidence, source)
        self.relations.append(r)
        self.fact_history.append({"type": "relation_added", "relation": asdict(r)})

    def remove_relation(self, subject: str, predicate: str, object_: str):
        self.relations = [
            r for r in self.relations
            if not (
                r.subject == subject
                and r.predicate == predicate
                and r.object == object_
            )
        ]

    def has_relation(
        self,
        subject: str,
        predicate: str,
        object_: Optional[str] = None,
    ) -> bool:
        return any(
            r.subject == subject
            and r.predicate == predicate
            and (object_ is None or r.object == object_)
            for r in self.relations
        )

    def query(
        self,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        object_: Optional[str] = None,
    ) -> List[Relation]:
        return [
            r for r in self.relations
            if (subject is None or r.subject == subject)
            and (predicate is None or r.predicate == predicate)
            and (object_ is None or r.object == object_)
        ]

    def entities_by_type(self, entity_type: str) -> List[Entity]:
        return [
            e for e in self.entities.values()
            if e.entity_type == entity_type
        ]

    def get_entity(self, entity_id: str) -> Entity:
        return self.entities[entity_id]

    def snapshot(self) -> Dict[str, Any]:
        return {
            "entities": {k: asdict(v) for k, v in self.entities.items()},
            "relations": [asdict(r) for r in self.relations],
        }

    def print_state(self):
        print("\n========== KNOWLEDGE GRAPH ==========")
        for e in self.entities.values():
            if e.entity_type in {"object", "surface", "human", "container"}:
                print(e.entity_id, "|", e.entity_type, "|",
                      e.label, "|", e.properties)
        print("\nRelations:")
        for r in self.relations:
            print(
                f"{r.subject} --{r.predicate}--> {r.object} "
                f"[{r.confidence:.2f}; {r.source}]"
            )


# ============================================================================
# 6. PERCEPTION
# ============================================================================

class YOLOPerception:
    """
    Replace detect() with the deployed YOLO + camera/ROS pipeline.

    The simulated observation deliberately creates the dynamic information
    during operation rather than pre-populating the KG.
    """

    def __init__(self, model_name: str = "yolo11n.pt"):
        self.model_name = model_name

    def detect(self, frame: Any = None) -> List[Detection]:
        if EXECUTION_MODE == ExecutionMode.REAL_ROBOT:
            return self._real_detection(frame)

        return [
            Detection(
                "det_cup_1", "cup", 0.96,
                BoundingBox(450, 260, 520, 340),
                Pose3D(1.15, 0.22, 0.84),
                "YOLO",
            ),
            Detection(
                "det_table_1", "table", 0.98,
                BoundingBox(300, 180, 700, 600),
                Pose3D(1.20, 0.20, 0.72),
                "YOLO",
            ),
            Detection(
                "det_person_1", "person", 0.99,
                BoundingBox(750, 100, 980, 600),
                Pose3D(0.10, 0.00, 1.10),
                "YOLO",
            ),
        ]

    def _real_detection(self, frame: Any) -> List[Detection]:
        raise NotImplementedError(
            "Connect this adapter to the deployed YOLO/ROS camera pipeline."
        )


class MiDaSDepthEstimator:
    """MiDaS depth estimation interface."""

    def estimate_depth(
        self,
        frame: Any,
        detections: List[Detection],
    ) -> List[Detection]:
        # Real deployment:
        # frame -> MiDaS -> depth map -> depth at each detection.
        return detections


class CoordinateTransformer:
    """Camera/image frame -> robot/world frame."""

    def image_to_robot(self, detection: Detection) -> Pose3D:
        if detection.pose_3d is None:
            raise ValueError("3D target position is unavailable.")
        # Real deployment:
        # p_robot = T_robot_camera @ p_camera
        return detection.pose_3d


# ============================================================================
# 7. SYMBOLIC SPATIAL GROUNDING
# ============================================================================

class SpatialReasoner:

    @staticmethod
    def iou(a: BoundingBox, b: BoundingBox) -> float:
        left = max(a.x1, b.x1)
        top = max(a.y1, b.y1)
        right = min(a.x2, b.x2)
        bottom = min(a.y2, b.y2)

        intersection = (
            max(0.0, right - left) *
            max(0.0, bottom - top)
        )
        union = a.area + b.area - intersection
        return intersection / union if union > 0 else 0.0

    def infer_on_relation(
        self,
        object_detection: Detection,
        surface_detection: Detection,
    ) -> Tuple[bool, float, str]:

        cx, cy = object_detection.bbox.center

        horizontal_containment = (
            surface_detection.bbox.x1 <= cx <= surface_detection.bbox.x2
        )

        vertical_consistency = (
            object_detection.bbox.center[1]
            >= surface_detection.bbox.y1
        )

        overlap = self.iou(
            object_detection.bbox,
            surface_detection.bbox,
        )

        # In production, combine segmentation/3D surface geometry with
        # calibrated camera information. This is a symbolic predicate whose
        # truth value is grounded in perception.
        score = (
            0.50 * float(horizontal_containment)
            + 0.25 * float(vertical_consistency)
            + 0.25 * overlap
        )

        result = (
            horizontal_containment
            and vertical_consistency
            and score >= CONTAINMENT_THRESHOLD
        )

        explanation = (
            f"center=({cx:.1f},{cy:.1f}), "
            f"horizontal_containment={horizontal_containment}, "
            f"vertical_consistency={vertical_consistency}, "
            f"IoU={overlap:.3f}, score={score:.3f}"
        )
        return result, score, explanation

    @staticmethod
    def distance(a: Pose3D, b: Pose3D) -> float:
        return math.sqrt(
            (a.x - b.x) ** 2 +
            (a.y - b.y) ** 2 +
            (a.z - b.z) ** 2
        )


# ============================================================================
# 8. SCENE GROUNDER — PERCEPTION -> DYNAMIC KG
# ============================================================================

class SceneGrounder:
    def __init__(
        self,
        kg: KnowledgeGraph,
        spatial_reasoner: SpatialReasoner,
        depth: MiDaSDepthEstimator,
        transformer: CoordinateTransformer,
    ):
        self.kg = kg
        self.spatial = spatial_reasoner
        self.depth = depth
        self.transformer = transformer

    def instantiate(
        self,
        detections: List[Detection],
        frame: Any = None,
    ) -> Dict[str, str]:

        detections = self.depth.estimate_depth(frame, detections)
        entity_map: Dict[str, str] = {}

        for d in detections:
            label = d.label.lower()
            if label in {"person", "human"}:
                entity_type = "human"
            elif label in {"table", "desk", "counter"}:
                entity_type = "surface"
            else:
                entity_type = "object"

            entity_id = f"{entity_type}_{d.detection_id}"
            props = {
                "bbox": asdict(d.bbox),
                "confidence": d.confidence,
                "visible": True,
                "detected_by": d.source_model,
            }

            if d.pose_3d is not None:
                props["position_3d"] = asdict(
                    self.transformer.image_to_robot(d)
                )

            if entity_type == "object":
                props.update({"movable": True, "graspable": True})

            self.kg.add_entity(Entity(
                entity_id, entity_type, d.label, props
            ))

            # Model itself can be represented as a KG entity.
            model_id = f"model_{d.source_model.lower().replace(' ', '_')}"
            if model_id not in self.kg.entities:
                self.kg.add_entity(Entity(
                    model_id, "model", d.source_model
                ))

            self.kg.add_relation(
                entity_id,
                "detected_by",
                model_id,
                d.confidence,
                "perception",
            )
            entity_map[d.detection_id] = entity_id

        objects = [
            d for d in detections
            if d.label.lower() not in {"person", "human", "table", "desk", "counter"}
        ]
        surfaces = [
            d for d in detections
            if d.label.lower() in {"table", "desk", "counter"}
        ]

        for obj in objects:
            for surface in surfaces:
                is_on, score, explanation = self.spatial.infer_on_relation(
                    obj, surface
                )
                if is_on:
                    obj_id = entity_map[obj.detection_id]
                    surface_id = entity_map[surface.detection_id]

                    self.kg.add_relation(
                        obj_id,
                        "on",
                        surface_id,
                        score,
                        "geometric_symbolic_reasoning",
                    )
                    self.kg.fact_history.append({
                        "type": "derived_fact",
                        "predicate": "on",
                        "subject": obj_id,
                        "object": surface_id,
                        "evidence": explanation,
                    })

        return entity_map


# ============================================================================
# 9. NAVIGATION FEASIBILITY
# ============================================================================

class NavigationPlanner:
    def check_path(
        self,
        start: Pose3D,
        destination: Pose3D,
    ) -> ConstraintResult:

        distance = SpatialReasoner.distance(start, destination)

        # Replace with SLAM/map + planner + obstacle/costmap in deployment.
        exists = distance < 5.0

        return ConstraintResult(
            "navigation_path_exists",
            exists,
            f"distance={distance:.2f} m; path_exists={exists}",
            {"distance_m": distance},
        )


# ============================================================================
# 10. IK / KINEMATIC FEASIBILITY
# ============================================================================

@dataclass
class IKResult:
    success: bool
    joint_configuration: Optional[List[float]]
    reason: str


class IKSolver:
    def solve(self, target_pose: Pose3D) -> IKResult:
        # Replace with Pepper/MoveIt/NAOqi IK in deployment.
        reachable = (
            -0.5 <= target_pose.x <= 2.0
            and -1.0 <= target_pose.y <= 1.0
            and 0.25 <= target_pose.z <= 1.60
        )

        if not reachable:
            return IKResult(
                False, None,
                "Target lies outside the configured robot workspace."
            )

        q = [0.20, -0.45, 0.75, -0.80, 0.30, 0.10]
        return IKResult(True, q, "Valid IK solution found.")

    def validate_joint_limits(
        self,
        q: List[float],
    ) -> ConstraintResult:

        limits = [
            (-2.0, 2.0),
            (-1.5, 1.5),
            (-2.0, 2.0),
            (-2.0, 2.0),
            (-1.5, 1.5),
            (-2.0, 2.0),
        ]

        valid = all(
            low <= value <= high
            for value, (low, high) in zip(q, limits)
        )

        return ConstraintResult(
            "joint_limits_valid",
            valid,
            "IK configuration satisfies all joint limits."
            if valid else
            "IK configuration violates at least one joint limit.",
            {"q": q, "limits": limits},
        )


class CollisionChecker:
    def check(
        self,
        target_pose: Pose3D,
        q: List[float],
    ) -> ConstraintResult:
        # Replace with trajectory-level collision checking against the
        # environment map and robot self-collision model.
        collision_free = True

        return ConstraintResult(
            "collision_free",
            collision_free,
            "Candidate trajectory is collision-free."
            if collision_free else
            "Candidate trajectory intersects an obstacle.",
            {"clearance_m": 0.12},
        )


class GraspPlanner:
    def check_geometry(self, object_entity: Entity) -> ConstraintResult:
        feasible = bool(
            object_entity.properties.get("graspable", False)
        )
        return ConstraintResult(
            "grasp_geometry_feasible",
            feasible,
            "Object is compatible with the configured grasp model."
            if feasible else
            "Object is not currently graspable.",
            {"object": object_entity.entity_id},
        )


# ============================================================================
# 11. ORCHESTRATION PLAN
# ============================================================================

@dataclass
class PlanStep:
    step_id: str
    action: str
    target: Optional[str] = None
    required_model: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OrchestrationPlan:
    plan_id: str
    goal: str
    steps: List[PlanStep]
    status: str = "candidate"
    reasoning: str = ""

    def to_dict(self):
        return {
            "plan_id": self.plan_id,
            "goal": self.goal,
            "status": self.status,
            "reasoning": self.reasoning,
            "steps": [asdict(s) for s in self.steps],
        }


# ============================================================================
# 12. POST-TRAINED GEMINI — NEURAL REASONING
# ============================================================================

class PostTrainedGeminiOrchestrator:
    """
    The actual deployment connects this class to the final Gemini adapter
    produced by the previous post-training + EWC stage.

    Conceptual interface:
        Gemini + final LoRA
          -> user request + NP output
          -> candidate orchestration

    The first orchestration is deliberately NOT considered executable.
    """

    def __init__(
        self,
        model_name: str = GEMINI_MODEL,
        adapter: str = "final_lora_adapter",
    ):
        self.model_name = model_name
        self.adapter = adapter

    def generate_initial_plan(
        self,
        user_request: str,
        np_description: Dict[str, Any],
    ) -> OrchestrationPlan:

        # Gemini orchestration interface:
        #
        # response = gemini.generate_orchestration(
        #     model=self.model_name,
        #     adapter=self.adapter,
        #     prompt=user_request,
        #     np_description=np_description,
        #     mode="candidate_only",
        # )
        #
        # Parse response -> OrchestrationPlan.

        # Deterministic implementation for the configured execution mode.
        request = user_request.lower()

        if "bring" in request and "cup" in request:
            steps = [
                PlanStep("s1", "detect", "cup", "YOLO"),
                PlanStep("s2", "locate", "cup", "Coordinate Transformation"),
                PlanStep("s3", "navigate", "cup"),
                PlanStep("s4", "grasp", "cup", "IK Solver"),
                PlanStep("s5", "navigate", "user"),
                PlanStep("s6", "give", "user"),
            ]
        else:
            steps = [
                PlanStep("s1", "detect", "requested_object", "YOLO"),
                PlanStep(
                    "s2", "locate", "requested_object",
                    "Coordinate Transformation"
                ),
                PlanStep("s3", "navigate", "requested_object"),
            ]

        return OrchestrationPlan(
            str(uuid.uuid4()),
            user_request,
            steps,
            "candidate",
            "Candidate generated by post-trained Gemini; "
            "symbolic validation is still required.",
        )

    def revise_plan(
        self,
        user_request: str,
        previous_plan: OrchestrationPlan,
        report: ValidationReport,
        kg_context: Dict[str, Any],
        np_description: Dict[str, Any],
    ) -> OrchestrationPlan:

        # Gemini orchestration interface:
        #
        # response = gemini.replan(
        #     model=self.model_name,
        #     adapter=self.adapter,
        #     prompt=user_request,
        #     candidate_plan=previous_plan.to_dict(),
        #     kg_context=kg_context,
        #     validation_report=report.summary(),
        #     np_description=np_description,
        # )

        failed = {
            c.name for c in report.checks if not c.passed
        }

        # Example symbolic-feedback-driven re-planning.
        steps = previous_plan.steps

        if {"ik_solution_exists", "target_reachable"} & failed:
            steps = [
                PlanStep("r1", "detect", "cup", "YOLO"),
                PlanStep("r2", "locate", "cup", "Coordinate Transformation"),
                PlanStep("r3", "approach", "cup"),
                PlanStep("r4", "grasp", "cup", "IK Solver"),
                PlanStep("r5", "navigate", "user"),
                PlanStep("r6", "give", "user"),
            ]

        return OrchestrationPlan(
            str(uuid.uuid4()),
            user_request,
            steps,
            "candidate",
            "Candidate revised by Gemini using KG-derived validation feedback.",
        )


# ============================================================================
# 13. SYMBOLIC PLAN VALIDATOR
# ============================================================================

class SymbolicPlanValidator:
    def __init__(
        self,
        kg: KnowledgeGraph,
        navigation: NavigationPlanner,
        ik: IKSolver,
        collision: CollisionChecker,
        grasp: GraspPlanner,
    ):
        self.kg = kg
        self.navigation = navigation
        self.ik = ik
        self.collision = collision
        self.grasp = grasp

    def find_object(self, label: str) -> Optional[Entity]:
        for e in self.kg.entities_by_type("object"):
            if e.label.lower() == label.lower():
                return e
        return None

    def find_human(self) -> Optional[Entity]:
        humans = self.kg.entities_by_type("human")
        return humans[0] if humans else None

    def validate(
        self,
        plan: OrchestrationPlan,
    ) -> ValidationReport:

        checks: List[ConstraintResult] = []
        missing: List[str] = []

        # --------------------------------------------------------------------
        # Rule: candidate plans cannot be executed.
        # --------------------------------------------------------------------
        checks.append(ConstraintResult(
            "execution_requires_validated_plan",
            plan.status != "executed",
            "Plan is still in candidate/validation state."
        ))

        # --------------------------------------------------------------------
        # Target resolution.
        # --------------------------------------------------------------------
        target_step = next(
            (
                s for s in plan.steps
                if s.target
                and s.target not in {"user", "requested_object"}
                and s.action in {"detect", "locate", "grasp", "pick"}
            ),
            None,
        )

        target = (
            self.find_object(target_step.target)
            if target_step else None
        )

        if target is None:
            checks.append(ConstraintResult(
                "object_exists",
                False,
                "Target object has not been grounded in the current scene."
            ))
            missing.append("target_object")
            return ValidationReport(
                False, checks, missing, True
            )

        checks.append(ConstraintResult(
            "object_exists",
            True,
            f"Target {target.entity_id} was grounded by perception."
        ))

        # --------------------------------------------------------------------
        # Spatial relation grounding.
        # --------------------------------------------------------------------
        if "on" in plan.goal.lower():
            surfaces = [
                e for e in self.kg.entities_by_type("surface")
                if e.label.lower() in {"table", "desk", "counter"}
            ]
            relation_verified = any(
                self.kg.has_relation(target.entity_id, "on", s.entity_id)
                for s in surfaces
            )

            checks.append(ConstraintResult(
                "requested_spatial_relation",
                relation_verified,
                "Requested object-surface relation is geometrically grounded."
                if relation_verified else
                "Object is not verified as being on the requested surface."
            ))

            if not relation_verified:
                missing.append("verified_spatial_relation")

        # --------------------------------------------------------------------
        # Coordinates.
        # --------------------------------------------------------------------
        pose_dict = target.properties.get("position_3d")
        if pose_dict is None:
            checks.append(ConstraintResult(
                "target_coordinates_available",
                False,
                "No 3D robot-frame target coordinates are available."
            ))
            missing.append("target_coordinates")
            return ValidationReport(False, checks, missing, True)

        checks.append(ConstraintResult(
            "target_coordinates_available",
            True,
            "3D robot-frame target coordinates are available.",
            {"position_3d": pose_dict},
        ))

        target_pose = Pose3D(**pose_dict)

        # --------------------------------------------------------------------
        # IK.
        # --------------------------------------------------------------------
        ik_result = self.ik.solve(target_pose)
        checks.append(ConstraintResult(
            "ik_solution_exists",
            ik_result.success,
            ik_result.reason,
            {"q": ik_result.joint_configuration},
        ))

        if not ik_result.success:
            missing.append("feasible_ik_configuration")
            return ValidationReport(False, checks, missing, True)

        # --------------------------------------------------------------------
        # Joint limits.
        # --------------------------------------------------------------------
        joint_check = self.ik.validate_joint_limits(
            ik_result.joint_configuration
        )
        checks.append(joint_check)

        if not joint_check.passed:
            return ValidationReport(False, checks, missing, True)

        # --------------------------------------------------------------------
        # Collision / trajectory.
        # --------------------------------------------------------------------
        collision_check = self.collision.check(
            target_pose,
            ik_result.joint_configuration,
        )
        checks.append(collision_check)

        if not collision_check.passed:
            return ValidationReport(False, checks, missing, True)

        # --------------------------------------------------------------------
        # Grasp geometry.
        # --------------------------------------------------------------------
        grasp_check = self.grasp.check_geometry(target)
        checks.append(grasp_check)

        if not grasp_check.passed:
            return ValidationReport(False, checks, missing, True)

        # --------------------------------------------------------------------
        # Navigation.
        # --------------------------------------------------------------------
        pepper = self.kg.get_entity("pepper")
        start_pose = Pose3D(**pepper.properties.get(
            "position_3d",
            {"x": 0.0, "y": 0.0, "z": 1.0},
        ))

        nav_to_object = self.navigation.check_path(
            start_pose, target_pose
        )
        checks.append(nav_to_object)

        if not nav_to_object.passed:
            return ValidationReport(False, checks, missing, True)

        # --------------------------------------------------------------------
        # Human delivery feasibility.
        # --------------------------------------------------------------------
        if any(
            s.action in {"give", "deliver"}
            for s in plan.steps
        ):
            human = self.find_human()

            if human is None:
                checks.append(ConstraintResult(
                    "human_target_known",
                    False,
                    "No human target is grounded in the current scene."
                ))
                missing.append("human_target")
            else:
                checks.append(ConstraintResult(
                    "human_target_known",
                    True,
                    f"Human {human.entity_id} is grounded."
                ))

                hpose_dict = human.properties.get("position_3d")
                if hpose_dict:
                    hpose = Pose3D(**hpose_dict)
                    delivery = self.navigation.check_path(
                        target_pose, hpose
                    )
                    delivery.name = "human_reachable"
                    checks.append(delivery)

                    if not delivery.passed:
                        missing.append("reachable_human")

        valid = all(c.passed for c in checks) and not missing

        return ValidationReport(
            valid,
            checks,
            missing,
            not valid,
        )


# ============================================================================
# 14. NP INTERFACE — PREVIOUS STAGE
# ============================================================================

class NeuralProcessOutput:
    """
    NP answers WHAT technical components/actions are needed.
    Gemini answers HOW they should be chronologically orchestrated.
    """

    @staticmethod
    def predict(user_request: str) -> Dict[str, Any]:
        # Replace with the saved NP meta-learner from the previous stage.
        return {
            "models": [
                "YOLO",
                "Coordinate Transformation",
                "IK Solver",
                "Navigation Planner",
            ],
            "actions": [
                "detect",
                "locate",
                "navigate",
                "grasp",
                "give",
            ],
        }


# ============================================================================
# 15. PEPPER EXECUTION
# ============================================================================

class PepperExecutor:
    """
    Safety boundary: only validated plans can reach this class.
    """

    def __init__(self, kg: KnowledgeGraph):
        self.kg = kg

    def execute(self, plan: OrchestrationPlan):
        if plan.status != "validated":
            raise RuntimeError(
                "Execution blocked: orchestration has not been validated."
            )

        print("\n========== PEPPER EXECUTION ==========")

        for step in plan.steps:
            print(
                f"{step.step_id}: {step.action}"
                + (f" -> {step.target}" if step.target else "")
                + (
                    f" [{step.required_model}]"
                    if step.required_model else ""
                )
            )

            self.execute_step(step)

        plan.status = "executed"

        self.kg.update_entity_properties(
            "pepper",
            {"execution_state": "completed"},
        )

    def execute_step(self, step: PlanStep):
        # Replace branches with ROS/NAOqi calls.
        if step.action in {
            "detect", "locate", "navigate", "approach",
            "grasp", "pick", "place", "release",
            "give", "deliver", "speak",
        }:
            return

        raise ValueError(f"Unsupported action: {step.action}")


# ============================================================================
# 16. COMPLETE NEURO-SYMBOLIC SYSTEM
# ============================================================================

class NeuroSymbolicPepperSystem:

    def __init__(self):
        # ---------------------------- Symbolic layer ------------------------
        self.schema = KnowledgeGraphSchema()
        self.kg = KnowledgeGraph(self.schema)

        # ---------------------------- Perception ----------------------------
        self.yolo = YOLOPerception()
        self.midas = MiDaSDepthEstimator()
        self.transformer = CoordinateTransformer()
        self.spatial = SpatialReasoner()

        self.grounder = SceneGrounder(
            self.kg,
            self.spatial,
            self.midas,
            self.transformer,
        )

        # ---------------------------- Feasibility ---------------------------
        self.navigation = NavigationPlanner()
        self.ik = IKSolver()
        self.collision = CollisionChecker()
        self.grasp = GraspPlanner()

        # ---------------------------- Neural layer --------------------------
        self.gemini = PostTrainedGeminiOrchestrator()

        # ---------------------------- Validation ----------------------------
        self.validator = SymbolicPlanValidator(
            self.kg,
            self.navigation,
            self.ik,
            self.collision,
            self.grasp,
        )

        self.executor = PepperExecutor(self.kg)

    def run(
        self,
        user_request: str,
        frame: Any = None,
    ) -> OrchestrationPlan:

        print("\n================================================")
        print("NEURO-SYMBOLIC PEPPER ORCHESTRATION")
        print("================================================")

        # ====================================================================
        # STAGE A — NP technical description
        # ====================================================================
        np_output = NeuralProcessOutput.predict(user_request)

        print("\n[A] NP OUTPUT — TECHNICAL REQUIREMENTS")
        print(json.dumps(np_output, indent=2))

        # ====================================================================
        # STAGE B — POST-TRAINED GEMINI INITIAL PLAN
        # ====================================================================
        candidate = self.gemini.generate_initial_plan(
            user_request,
            np_output,
        )

        # Critical: candidate != validated plan.
        candidate.status = "candidate"

        print("\n[B] INITIAL GEMINI ORCHESTRATION — NOT EXECUTABLE")
        print(json.dumps(candidate.to_dict(), indent=2))

        # ====================================================================
        # STAGE C — OBSERVE ENVIRONMENT
        # ====================================================================
        print("\n[C] PERCEPTION")

        detections = self.yolo.detect(frame)

        for d in detections:
            print(
                f"YOLO -> {d.label} "
                f"confidence={d.confidence:.2f}, "
                f"bbox={asdict(d.bbox)}, "
                f"pose={asdict(d.pose_3d) if d.pose_3d else None}"
            )

        # ====================================================================
        # STAGE D — DYNAMIC KG INSTANTIATION
        # ====================================================================
        print("\n[D] PERCEPTION -> DYNAMIC KNOWLEDGE GRAPH")

        self.grounder.instantiate(detections, frame)
        self.kg.print_state()

        # ====================================================================
        # STAGE E/F — SYMBOLIC VALIDATION + NEURAL REPLANNING
        # ====================================================================
        current_plan = candidate

        for round_id in range(1, MAX_REPLANNING_ROUNDS + 1):

            print(
                f"\n[E] SYMBOLIC VALIDATION — ROUND {round_id}"
            )

            report = self.validator.validate(current_plan)
            print(report.summary())

            if report.valid:
                current_plan.status = "validated"
                break

            if round_id == MAX_REPLANNING_ROUNDS:
                raise RuntimeError(
                    "No validated orchestration was obtained after "
                    "the maximum replanning rounds."
                )

            print(
                "\n[F] GEMINI REPLANNING FROM SYMBOLIC FEEDBACK"
            )

            current_plan = self.gemini.revise_plan(
                user_request,
                current_plan,
                report,
                self.kg.snapshot(),
                np_output,
            )

        # ====================================================================
        # STAGE G — EXECUTION ONLY AFTER VALIDATION
        # ====================================================================
        print("\n[G] FINAL VALIDATED ORCHESTRATION")
        print(json.dumps(current_plan.to_dict(), indent=2))

        self.executor.execute(current_plan)

        # ====================================================================
        # STAGE H — DYNAMIC KG UPDATE
        # ====================================================================
        self.kg.update_entity_properties(
            "pepper",
            {"execution_state": "idle"},
        )

        print("\n[H] KG UPDATED AFTER EXECUTION")
        self.kg.print_state()

        return current_plan


# ============================================================================
# 17. GENERAL SYMBOLIC RULE ENGINE
# ============================================================================

class RuleEngine:
    """
    Extension point for additional symbolic predicates and constraints.

    Examples:
        - object must exist
        - robot must support an action
        - action dependencies must be respected
        - holding required before delivery
        - destination must be reachable
        - object must not be obstructed
    """

    def __init__(self, kg: KnowledgeGraph):
        self.kg = kg

    def object_exists(self, label: str) -> bool:
        return any(
            e.label.lower() == label.lower()
            for e in self.kg.entities_by_type("object")
        )

    def robot_supports_action(self, action: str) -> bool:
        pepper = self.kg.get_entity("pepper")
        capabilities = pepper.properties.get("capabilities", {})

        mapping = {
            "navigate": "navigation",
            "detect": "visual_perception",
            "grasp": "grasping",
            "give": "arm_manipulation",
            "speak": "speech_synthesis",
        }

        capability = mapping.get(action)
        return bool(capability and capabilities.get(capability, False))

    def validate_dependencies(
        self,
        plan: OrchestrationPlan,
    ) -> List[ConstraintResult]:

        results = []
        seen: Set[str] = set()

        for step in plan.steps:

            if step.action in {"grasp", "pick"}:
                ok = "detect" in seen and "locate" in seen
                results.append(ConstraintResult(
                    f"{step.step_id}_perception_precondition",
                    ok,
                    "Detection and localization precede manipulation."
                    if ok else
                    "Manipulation appears before required perception."
                ))

            if step.action in {"give", "deliver"}:
                ok = "grasp" in seen or "pick" in seen
                results.append(ConstraintResult(
                    f"{step.step_id}_holding_precondition",
                    ok,
                    "Object acquisition precedes delivery."
                    if ok else
                    "Delivery appears before object acquisition."
                ))

            seen.add(step.action)

        return results


# ============================================================================
# 18. SERIALIZATION
# ============================================================================

def save_kg(kg: KnowledgeGraph, path: str = "pepper_dynamic_kg.json"):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(kg.snapshot(), f, indent=2)
    print(f"KG saved to: {path}")


# ============================================================================
# 19. DEMONSTRATION
# ============================================================================

def main():
    system = NeuroSymbolicPepperSystem()

    user_request = "Give me the cup on the table."

    final_plan = system.run(
        user_request=user_request,
        frame=None,
    )

    print("\n================================================")
    print("FINAL RESULT")
    print("================================================")
    print(json.dumps(final_plan.to_dict(), indent=2))

    save_kg(system.kg)


if __name__ == "__main__":
    main()
