use rust_tensor_einsum::Tensor;

fn main() {
    run_case1_transitivity();
    run_case2_syllogism();
    run_case3_contextual();
}

fn run_case1_transitivity() {
    println!("\n=== Case 1: Transitivity (推移律) ===");
    println!("Goal: A x B represents logical transitivity A->B->C.");
    println!("Setting: Partial Order (1 <= 2, 2 <= 3)");

    let n = 3;
    let mut data = vec![vec![0.0; n]; n];
    
    // 1 <= 2, 2 <= 3
    data[0][1] = 1.0;
    data[1][2] = 1.0;

    let r = Tensor::from_vec2(data);
    println!("Input: Relation Matrix R (1<=2, 2<=3)");
    r.print_nonzero();

    // Export Input (入力のエクスポート)
    let labels = &["1", "2", "3"];
    r.export_relation_csv(
        "tensor_case1_input.csv",
        &["LHS", "RHS", "Value"],
        &[labels, labels]
    );

    // Logical Inference: R x R (論理的推論: R x R)
    let r2 = Tensor::einsum("ij,jk->ik", &[&r, &r]);
    
    println!("Output: Derived Relation (1<=3 is derived)");
    r2.print_nonzero();

    // Export Output (出力のエクスポート)
    r2.export_relation_csv(
        "tensor_case1_output.csv",
        &["LHS", "RHS", "Value"],
        &[labels, labels]
    );
}

fn run_case2_syllogism() {
    println!("\n=== Case 2: Syllogism and Cut Elimination (三段論法とカット除去) ===");
    println!("Goal: Tensor contraction as Cut Elimination.");
    println!("Premise: 'Socrates is Human', 'Human implies Mortal' => 'Socrates is Mortal'");
    
    let subject_labels = &["Socrates", "Zeus"];
    let concept_labels = &["Human", "God"];
    let quality_labels = &["Mortal", "Immortal"];

    // Fact: Socrates is Human, Zeus is God (事実: ソクラテスは人間、ゼウスは神)
    let facts_data = vec![
        vec![1.0, 0.0], // Socrates -> Human (ソクラテス -> 人間)
        vec![0.0, 1.0], // Zeus -> God (ゼウス -> 神)
    ];
    let facts = Tensor::from_vec2(facts_data);

    // Rule: Human -> Mortal, God -> Immortal (ルール: 人間 -> 死すべきもの、神 -> 不死)
    let rules_data = vec![
        vec![1.0, 0.0], // Human -> Mortal (人間 -> 死すべきもの)
        vec![0.0, 1.0], // God -> Immortal (神 -> 不死)
    ];
    let rules = Tensor::from_vec2(rules_data);
    
    println!("Input: Facts and Rules");
    facts.print_nonzero();
    rules.print_nonzero();
    
    facts.export_relation_csv(
        "tensor_case2_facts.csv",
        &["Subject", "Concept", "Value"],
        &[subject_labels, concept_labels]
    );
    rules.export_relation_csv(
        "tensor_case2_rules.csv",
        &["Concept", "Quality", "Value"],
        &[concept_labels, quality_labels]
    );

    // Cut Elimination: "Concept" is cut out (カット除去: "概念"項がカットされる).
    let conclusion = Tensor::einsum("sc,cq->sq", &[&facts, &rules]);

    println!("Output: Conclusion (Intermediate concept eliminated)");
    conclusion.print_nonzero();

    conclusion.export_relation_csv(
        "tensor_case2_conclusion.csv",
        &["Subject", "Quality", "Value"],
        &[subject_labels, quality_labels]
    );
}

fn run_case3_contextual() {
    println!("\n=== Case 3: Contextual Inference (文脈と推論) ===");
    println!("Goal: Representing Context as a tensor dimension.");
    println!("(Note: Names are anonymized for this public demo)");
    
    // Anonymized names (匿名化された名前)
    let people = &["Alice", "Bob", "Charlie"];
    let contexts = &["Official", "Private"];
    
    let n_p = 3;
    let n_c = 2;
    
    // R[x, y, c]
    let mut data = vec![vec![vec![0.0; n_c]; n_p]; n_p];
    
    // Official Context: Alice -> Bob -> Charlie (公式な文脈: Alice -> Bob -> Charlie)
    // Alice manages Bob, Bob manages Charlie (AliceはBobを管理し、BobはCharlieを管理する).
    data[0][1][0] = 1.0; // A->B
    data[1][2][0] = 1.0; // B->C
    
    // Private Context: Bob -> Charlie (Friends), Alice not involved (私的な文脈: Bob -> Charlie (友人)、Aliceは関与しない).
    data[1][2][1] = 1.0; // B->C
    
    let r = Tensor::from_vec3(data);
    println!("Input: Relation Tensor");
    r.print_nonzero();
    
    r.export_relation_csv(
        "tensor_case3_input.csv",
        &["Subject", "Object", "Context", "Value"],
        &[people, people, contexts]
    );

    // Inference preserving context (文脈を保存した推論)
    let r2 = Tensor::einsum("xyc,yzc->xzc", &[&r, &r]);
    
    println!("Output: Inference Result");
    println!("(Transitivity holds in Official context, but not necessarily in Private context)");
    r2.print_nonzero();
    
    r2.export_relation_csv(
        "tensor_case3_output.csv",
        &["Subject", "Object", "Context", "Value"],
        &[people, people, contexts]
    );
}
