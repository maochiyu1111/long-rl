当前仓库long-rl的当前分支diffusion是基于verl0.5，支持npu的版本，并且已经迁移支持了diffusion模型（视频/图片生成逻辑），但是RL算法并不是dancegrpo。
disco_rl的main分支是基于verl0.3，不支持npu的版本，并且RL算法也不是dancegrpo。long-rl的当前分支diffusion是从disco_rl的main分支迁移过来的，目的是为了支持npu，并且迁移diffusion相关的逻辑。
disco_rl的sijie/dancegrpo分支是基于verl0.3，不支持npu的版本，但是RL算法是dancegrpo。

我现在的目的是将dancegrpo的RL算法迁移到long-rl的当前分支diffusion中，并且支持npu。我已经做了一个分析，在`/Users/bytedance/codegfile/disco_rl/main-vs-dancegrpo-analysis.md`中。这个分析说明了disco_rl的main分支和dancegrpo分支的区别，以及迁移的影响。我现在需要你评估分析一下，将dancegrpo的RL算法迁移到long-rl的当前分支diffusion中应该采取什么策略？将分析报告输出到`transfer_analyze.md`当中



根据 `/Users/bytedance/codegfile/long-rl/transfer_strategy.md` 为阶段A：算法开关与配置骨架写一个函数级别细粒度的迁移todo表`transfer_A.md`，比如在xxx文件中增加xxx函数、或修改/新增xxx配置文件，增加xxx配置项。long-rl位于`/Users/bytedance/codegfile/long-rl`， disco_rl位于     迁移时做到尊重源代码，即尊重disco_rl中的代码逻辑，一定不要自我发挥，不要增加源代码中没有的函数，不要修改源代码中的代码逻辑。迁移todo表要能完整实现阶段目标。

根据 `/Users/bytedance/codegfile/long-rl/transfer_strategy.md` 为阶段B：Rollout 协议对齐（轨迹字段 + group 生成）写一个函数级别细粒度的迁移todo表`transfer_B.md`，比如在xxx文件中增加xxx函数、或修改/新增xxx配置文件，增加xxx配置项。long-rl位于`/Users/bytedance/codegfile/long-rl`， disco_rl位于`/Users/bytedance/codegfile/disco_rl` 迁移时做到尊重源代码，即尊重disco_rl中的代码逻辑，一定不要自我发挥，不要增加源代码中没有的函数，不要修改源代码中的代码逻辑。迁移todo表要能完整实现阶段目标。

根据 `/Users/bytedance/codegfile/long-rl/transfer_strategy.md` 为阶段C：奖励协议与双路优势写一个函数级别细粒度的迁移todo表`transfer_c.md` ，比如在xxx文件中增加xxx函数、或修改/新增xxx配置文件，增加xxx配置项。long-rl位于`/Users/bytedance/codegfile/long-rl`， disco_rl位于`/Users/bytedance/codegfile/disco_rl` 迁移时做到尊重源代码，即尊重disco_rl中的代码逻辑，一定不要自我发挥，不要增加源代码中没有的函数，不要修改源代码中的代码逻辑。迁移todo表要能完整实现阶段目标。

根据 `/Users/bytedance/codegfile/long-rl/transfer_strategy.md` 为阶段D：Dance 更新主路径（Best-of-N + step 子采样 + 双损失）写一个函数级别细粒度的迁移todo表`transfer_D.md`，比如在xxx文件中增加xxx函数、或修改/新增xxx配置文件，增加xxx配置项。long-rl位于`/Users/bytedance/codegfile/long-rl`， disco_rl位于`/Users/bytedance/codegfile/disco_rl` 迁移时做到尊重源代码，即尊重disco_rl中的代码逻辑，一定不要自我发挥，不要增加源代码中没有的函数，不要修改源代码中的代码逻辑。迁移todo表要能完整实现阶段目标。

根据 `/Users/bytedance/codegfile/long-rl/transfer_strategy.md` 为阶段E：VideoAlign 与外部奖励后端插件化接入写一个函数级别细粒度的迁移todo表`transfer_E.md`，比如在xxx文件中增加xxx函数、或修改/新增xxx配置文件，增加xxx配置项。long-rl位于`/Users/bytedance/codegfile/long-rl`， disco_rl位于`/Users/bytedance/codegfile/disco_rl` 迁移时做到尊重源代码，即尊重disco_rl中的代码逻辑，一定不要自我发挥，不要增加源代码中没有的函数，不要修改源代码中的代码逻辑。迁移todo表要能完整实现阶段目标。

根据 `/Users/bytedance/codegfile/long-rl/transfer_strategy.md` 为阶段F：异步流水线对齐与性能优化写一个函数级别细粒度的迁移todo表`transfer_F.md`，比如在xxx文件中增加xxx函数、或修改/新增xxx配置文件，增加xxx配置项。long-rl位于`/Users/bytedance/codegfile/long-rl`， disco_rl位于`/Users/bytedance/codegfile/disco_rl` 迁移时做到尊重源代码，即尊重disco_rl中的代码逻辑，一定不要自我发挥，不要增加源代码中没有的函数，不要修改源代码中的代码逻辑。迁移todo表要能完整实现阶段目标。


你是一个代码迁移专家，你需要审核transfer_A.md 这个迁移todo表格是否符合`/Users/bytedance/codegfile/long-rl/transfer_strategy.md` 中对应阶段的目标，有没有遗漏的地方，有没有错误的地方，是否能完整实现目标。long-rl位于`/Users/bytedance/codegfile/long-rl`， disco_rl位于`/Users/bytedance/codegfile/disco_rl` 。迁移时需要做到尊重源代码，即尊重disco_rl中的代码逻辑。若todo表无问题则无需修改，若有问题则请修改文件。

你是一个代码迁移专家，你需要审核transfer_B.md 这个迁移todo表格是否符合`/Users/bytedance/codegfile/long-rl/transfer_strategy.md` 中对应阶段的目标，有没有遗漏的地方，有没有错误的地方，是否能完整实现目标。long-rl位于`/Users/bytedance/codegfile/long-rl`， disco_rl位于`/Users/bytedance/codegfile/disco_rl` 。迁移时需要做到尊重源代码，即尊重disco_rl中的代码逻辑。若todo表无问题则无需修改，若有问题则请修改文件。

你是一个代码迁移专家，你需要审核transfer_c.md 这个迁移todo表格是否符合`/Users/bytedance/codegfile/long-rl/transfer_strategy.md` 中对应阶段的目标，有没有遗漏的地方，有没有错误的地方，是否能完整实现目标。long-rl位于`/Users/bytedance/codegfile/long-rl`， disco_rl位于`/Users/bytedance/codegfile/disco_rl` 。迁移时需要做到尊重源代码，即尊重disco_rl中的代码逻辑。若todo表无问题则无需修改，若有问题则请修改文件。

你是一个代码迁移专家，你需要审核transfer_D.md 这个迁移todo表格是否符合`/Users/bytedance/codegfile/long-rl/transfer_strategy.md` 中对应阶段的目标，有没有遗漏的地方，有没有错误的地方，是否能完整实现目标。long-rl位于`/Users/bytedance/codegfile/long-rl`， disco_rl位于`/Users/bytedance/codegfile/disco_rl` 。迁移时需要做到尊重源代码，即尊重disco_rl中的代码逻辑。若todo表无问题则无需修改，若有问题则请修改文件。

你是一个代码迁移专家，你需要审核transfer_E.md 这个迁移todo表格是否符合`/Users/bytedance/codegfile/long-rl/transfer_strategy.md` 中对应阶段的目标，有没有遗漏的地方，有没有错误的地方，是否能完整实现目标。long-rl位于`/Users/bytedance/codegfile/long-rl`， disco_rl位于`/Users/bytedance/codegfile/disco_rl` 。迁移时需要做到尊重源代码，即尊重disco_rl中的代码逻辑。若todo表无问题则无需修改，若有问题则请修改文件。

你是一个代码迁移专家，你需要审核transfer_F.md 这个迁移todo表格是否符合`/Users/bytedance/codegfile/long-rl/transfer_strategy.md` 中对应阶段的目标，有没有遗漏的地方，有没有错误的地方，是否能完整实现目标。long-rl位于`/Users/bytedance/codegfile/long-rl`， disco_rl位于`/Users/bytedance/codegfile/disco_rl` 。迁移时需要做到尊重源代码，即尊重disco_rl中的代码逻辑。若todo表无问题则无需修改，若有问题则请修改文件。