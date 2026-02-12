import random
import numpy as np # type: ignore
from collections import defaultdict
from sklearn.linear_model import LogisticRegression # pyright: ignore[reportMissingModuleSource]

# -----------------------------
# DATA DEFINITIONS
# -----------------------------

class Teacher:
    def __init__(self, name, subjects, max_per_week, availability):
        self.name = name
        self.subjects = subjects
        self.max_per_week = max_per_week
        self.availability = availability  # set of (day, period)
        self.assigned = []

class Subject:
    def __init__(self, name, lectures_per_week):
        self.name = name
        self.lectures_per_week = lectures_per_week

class Room:
    def __init__(self, name, capacity, availability):
        self.name = name
        self.capacity = capacity
        self.availability = availability

# -----------------------------
# TIMETABLE SYSTEM
# -----------------------------

class TimetableSystem:
    def __init__(self, teachers, subjects, rooms, days, periods):
        self.teachers = teachers
        self.subjects = subjects
        self.rooms = rooms
        self.days = days
        self.periods = periods
        self.timetable = defaultdict(dict)
        self.workload = defaultdict(int)

    # -------------------------
    # CONFLICT CHECK
    # -------------------------
    def has_conflict(self, teacher, room, day, period):
        slot = (day, period)

        # Teacher conflict
        for entry in self.timetable[day].values():
            if entry["teacher"] == teacher and entry["period"] == period:
                return True

        # Room conflict
        for entry in self.timetable[day].values():
            if entry["room"] == room and entry["period"] == period:
                return True

        # Availability
        if slot not in teacher.availability:
            return True

        if slot not in room.availability:
            return True

        return False

    # -------------------------
    # HEURISTIC SCORING
    # -------------------------
    def score_assignment(self, teacher, room, day, period):
        score = 0

        if (day, period) in teacher.availability:
            score += 10

        if (day, period) in room.availability:
            score += 10

        # Workload balancing
        avg_load = np.mean(list(self.workload.values())) if self.workload else 0
        score -= abs(self.workload[teacher.name] - avg_load)

        return score

    # -------------------------
    # GENERATE TIMETABLE
    # -------------------------
    def generate(self):
        for subject in self.subjects:
            lectures_needed = subject.lectures_per_week

            while lectures_needed > 0:
                best_score = -999
                best_choice = None

                for day in self.days:
                    for period in self.periods:
                        for teacher in self.teachers:
                            if subject.name not in teacher.subjects:
                                continue
                            if self.workload[teacher.name] >= teacher.max_per_week:
                                continue

                            for room in self.rooms:
                                if self.has_conflict(teacher, room, day, period):
                                    continue

                                score = self.score_assignment(teacher, room, day, period)

                                if score > best_score:
                                    best_score = score
                                    best_choice = (teacher, room, day, period)

                if best_choice:
                    teacher, room, day, period = best_choice
                    self.timetable[day][f"{subject.name}-{lectures_needed}"] = {
                        "subject": subject.name,
                        "teacher": teacher.name,
                        "room": room.name,
                        "period": period
                    }
                    self.workload[teacher.name] += 1
                    lectures_needed -= 1
                else:
                    print("Unable to assign:", subject.name)
                    break

    # -------------------------
    # SMART RESCHEDULING
    # -------------------------
    def reschedule_teacher(self, teacher_name):
        print(f"\nRescheduling for unavailable teacher: {teacher_name}")

        for day in self.days:
            for key, entry in list(self.timetable[day].items()):
                if entry["teacher"] == teacher_name:
                    subject_name = entry["subject"]
                    period = entry["period"]
                    room_name = entry["room"]

                    # Remove old assignment
                    del self.timetable[day][key]
                    self.workload[teacher_name] -= 1

                    # Try reassign
                    for teacher in self.teachers:
                        if teacher.name == teacher_name:
                            continue
                        if subject_name not in teacher.subjects:
                            continue

                        for room in self.rooms:
                            if not self.has_conflict(teacher, room, day, period):
                                self.timetable[day][key] = {
                                    "subject": subject_name,
                                    "teacher": teacher.name,
                                    "room": room.name,
                                    "period": period
                                }
                                self.workload[teacher.name] += 1
                                print("Reassigned successfully.")
                                return

        print("Could not reschedule.")

    # -------------------------
    # PRINT TIMETABLE
    # -------------------------
    def display(self):
        print("\nFINAL TIMETABLE\n")
        for day in self.days:
            print("Day:", day)
            for entry in self.timetable[day].values():
                print(entry)
            print()

# -----------------------------
# SEATING ARRANGEMENT
# -----------------------------

def generate_seating(students, rows, cols):
    random.shuffle(students)
    seating = []
    index = 0

    for r in range(rows):
        row = []
        for c in range(cols):
            if index < len(students):
                row.append(students[index])
                index += 1
            else:
                row.append("Empty")
        seating.append(row)

    return seating

# -----------------------------
# ML CONFLICT PREDICTION
# -----------------------------

def train_conflict_model():
    # Dummy historical data
    X = np.array([
        [1, 1],  # Monday 1st period
        [5, 6],  # Friday last period
        [3, 2],
        [2, 4],
        [5, 5]
    ])
    y = np.array([0, 1, 0, 0, 1])  # 1 = conflict

    model = LogisticRegression()
    model.fit(X, y)
    return model

# -----------------------------
# SAMPLE RUN
# -----------------------------

if __name__ == "__main__":

    days = ["Mon", "Tue", "Wed", "Thu", "Fri"]
    periods = [1, 2, 3, 4, 5]

    availability = {(d, p) for d in days for p in periods}

    teachers = [
        Teacher("T1", ["Math", "Physics"], 10, availability),
        Teacher("T2", ["Chemistry", "Math"], 10, availability),
        Teacher("T3", ["Biology"], 10, availability)
    ]

    subjects = [
        Subject("Math", 4),
        Subject("Physics", 3),
        Subject("Chemistry", 3),
        Subject("Biology", 2)
    ]

    rooms = [
        Room("R1", 40, availability),
        Room("R2", 50, availability)
    ]

    system = TimetableSystem(teachers, subjects, rooms, days, periods)
    system.generate()
    system.display()

    # Reschedule example
    system.reschedule_teacher("T1")
    system.display()

    # Seating Example
    students = [f"S{i}" for i in range(1, 21)]
    seating = generate_seating(students, 4, 5)
    print("\nSeating Arrangement:")
    for row in seating:
        print(row)

    # ML Model
    model = train_conflict_model()
    prediction = model.predict([[5, 6]])
    print("\nConflict Prediction for Friday 6th Period:", prediction[0])