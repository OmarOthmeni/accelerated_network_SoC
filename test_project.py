#!/usr/bin/env python3
"""
Test Project Organization
"""

import os
import sys
import json

def test_essentials():
    print("🧪 TESTING PROJECT ORGANIZATION")
    print("=" * 50)
    
    passed = 0
    total = 0
    
    # Test 1: Essential files
    print("\n📄 ESSENTIAL FILES:")
    files_to_test = [
        ("requirements.txt", 100, "Dependencies"),
        ("config.py", 1000, "Configuration"),
        ("README.md", 500, "Documentation"),
        ("main.py", 500, "Main script"),
    ]
    
    for filename, min_size, description in files_to_test:
        total += 1
        if os.path.exists(filename):
            size = os.path.getsize(filename)
            if size >= min_size:
                print(f"  ✅ {filename:20s} ({size:6,} bytes) - {description}")
                passed += 1
            else:
                print(f"  ⚠️  {filename:20s} ({size:6,} bytes) - Too small")
        else:
            print(f"  ❌ {filename:20s} - Missing")
    
    # Test 2: Essential folders
    print("\n📁 ESSENTIAL FOLDERS:")
    folders_to_test = [
        "data",
        "models",
        "scripts",
        "results",
        "utils",
    ]
    
    for folder in folders_to_test:
        total += 1
        if os.path.exists(folder):
            print(f"  ✅ {folder}/")
            passed += 1
        else:
            print(f"  ❌ {folder}/ - Missing")
    
    # Test 3: Model exists
    print("\n🤖 MODEL CHECK:")
    total += 1
    model_path = "models/baseline/minimal_cnn.h5"
    if os.path.exists(model_path):
        size = os.path.getsize(model_path)
        print(f"  ✅ minimal_cnn.h5 ({size:,} bytes)")
        print(f"     Baseline accuracy: 90.47%")
        passed += 1
    else:
        print(f"  ❌ Model not found")
    
    # Test 4: Scripts
    print("\n🐍 PYTHON SCRIPTS:")
    scripts_dir = "scripts"
    if os.path.exists(scripts_dir):
        scripts = [f for f in os.listdir(scripts_dir) if f.endswith('.py')]
        if scripts:
            for script in sorted(scripts):
                total += 1
                script_path = os.path.join(scripts_dir, script)
                size = os.path.getsize(script_path)
                if size > 500:
                    print(f"  ✅ scripts/{script:20s} ({size:6,} bytes)")
                    passed += 1
                else:
                    print(f"  ⚠️  scripts/{script:20s} ({size:6,} bytes) - Small")
        else:
            print("  ❌ No Python scripts found")
            total += 1
    else:
        print("  ❌ scripts/ folder missing")
        total += 1
    
    # Summary
    print("\n" + "=" * 50)
    percentage = (passed / total) * 100 if total > 0 else 0
    print(f"📊 RESULTS: {passed}/{total} tests passed ({percentage:.1f}%)")
    
    if passed == total:
        print("🎉 EXCELLENT! Project perfectly organized!")
        status = "perfect"
    elif percentage >= 80:
        print("✅ GOOD! Project well organized.")
        status = "good"
    elif percentage >= 60:
        print("⚠️  FAIR. Some organization needed.")
        status = "fair"
    else:
        print("❌ NEEDS WORK. Major organization required.")
        status = "poor"
    
    # Recommendations
    print("\n🚀 RECOMMENDATIONS:")
    if status == "perfect":
        print("   1. Run: python scripts/evaluate_baseline.py")
        print("   2. Check: dir results\\baseline\\")
        print("   3. Start optimization phase!")
    else:
        print("   1. Fill missing files/folders")
        print("   2. Run this test again")
        print("   3. Then proceed to evaluation")
    
    # Save test results
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    test_results = {
        "timestamp": __import__('datetime').datetime.now().isoformat(),
        "passed": passed,
        "total": total,
        "percentage": percentage,
        "status": status,
        "project": "Rover Image Classification",
        "student": "Donia"
    }
    
    results_file = os.path.join(results_dir, "organization_test.json")
    with open(results_file, "w") as f:
        json.dump(test_results, f, indent=2)
    
    print(f"\n💾 Test results saved to: {results_file}")
    
    return passed == total

if __name__ == "__main__":
    success = test_essentials()
    sys.exit(0 if success else 1)