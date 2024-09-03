%% 初始化环境与参数设置

% 初始化仿真时间
Ts = 1;  % 采样时间
T = 20;  % 总仿真时间

mdl = 'danche2758s'; % Simulink模型名称
agentblk = [mdl '/RL Agent']; % Simulink中RL Agent模块的路径

n = 3; % 只有后三辆车的加速度需要通过DDPG控制

% 状态空间模型参数
A = [0 1 0 0 0 0;
     0 0 0 0 0 0;
     0 0 0 1 0 0;
     0 0 0 0 0 0;
     0 0 0 0 0 1;
     0 0 0 0 0 0];

B = [0 0 0;
     1 0 0;
     0 0 0;
     0 1 0;
     0 0 0;
     0 0 1];

C = eye(6);  % 输出状态矩阵
D = zeros(6,3); % 没有直接传递输入到输出的关系

initialConditions = [ 10, 40, 0, 40, -10, 40 ]; % 初始条件设置


% 定义观测和动作的规格
observationInfo = rlNumericSpec([3 1],'LowerLimit',-inf*ones(3,1),'UpperLimit',inf*ones(3,1));
observationInfo.Name = 'observations';
observationInfo.Description = 'Vehicle Data Observations';

actionInfo = rlNumericSpec([n 1],'LowerLimit',-3*ones(n,1),'UpperLimit',3*ones(n,1)); % 动作空间设置
actionInfo.Name = 'actions';

% 定义Simulink环境
env = rlSimulinkEnv(mdl, agentblk, observationInfo, actionInfo);
env.ResetFcn = @(in)localResetFcn(in); % 设置重置函数
rng('default')  % 设置随机数种子，确保结果可重复

%% 定义Actor和Critic网络

L = 120; % 隐藏层神经元数量

% 定义Critic网络
statePath = [
    featureInputLayer(3, 'Normalization', 'none', 'Name', 'observation')
    fullyConnectedLayer(L, 'Name', 'fc1')
    reluLayer('Name', 'relu1')
    fullyConnectedLayer(L, 'Name', 'fc2')
    additionLayer(2, 'Name', 'add')
    reluLayer('Name', 'relu2')
    fullyConnectedLayer(L, 'Name', 'fc3')
    reluLayer('Name', 'relu3')
    fullyConnectedLayer(1, 'Name', 'fc4')]; % 输出为标量

actionPath = [
    featureInputLayer(n, 'Normalization', 'none', 'Name', 'action')
    fullyConnectedLayer(L, 'Name', 'fc5')];

criticNetwork = layerGraph(statePath);
criticNetwork = addLayers(criticNetwork, actionPath);
criticNetwork = connectLayers(criticNetwork, 'fc5', 'add/in2');
criticNetwork = dlnetwork(criticNetwork); % 将网络转换为深度学习网络
figure;
plot(layerGraph(criticNetwork)) % 可视化Critic网络
title('Critic Network')

criticOptions = rlOptimizerOptions('LearnRate',1e-3,'GradientThreshold',1,'L2RegularizationFactor',1e-4); %%For this you need Matlab 2022a

critic = rlQValueFunction(criticNetwork,observationInfo,actionInfo,...
    'ObservationInputNames','observation','ActionInputNames','action');

% 定义Actor网络
actorNetwork = [
    featureInputLayer(3, 'Normalization', 'none', 'Name', 'observation')
    fullyConnectedLayer(L, 'Name', 'fc1')
    reluLayer('Name', 'relu1')
    fullyConnectedLayer(L, 'Name', 'fc2')
    reluLayer('Name', 'relu2')
    fullyConnectedLayer(L, 'Name', 'fc3')
    reluLayer('Name', 'relu3')
    fullyConnectedLayer(3, 'Name', 'fc4') % 输出大小为3
    tanhLayer('Name', 'tanh1')
    scalingLayer('Name', 'ActorScaling1', 'Scale', 2.5*ones(3,1), 'Bias', -0.5*ones(3,1))];
actorNetwork = dlnetwork(actorNetwork);

figure;
plot(layerGraph(actorNetwork)) % 可视化Actor网络
title('Actor Network')

actorOptions = rlOptimizerOptions('LearnRate',1e-4,'GradientThreshold',1,'L2RegularizationFactor',1e-4);
actor = rlContinuousDeterministicActor(actorNetwork,observationInfo,actionInfo);
%% 定义DDPG Agent和训练设置
agentOptions = rlDDPGAgentOptions(...
    'SampleTime',Ts,...
    'ActorOptimizerOptions',actorOptions,...
    'CriticOptimizerOptions',criticOptions,...
    'ExperienceBufferLength',1e6);
agentOptions.NoiseOptions.Variance = 0.6;
agentOptions.NoiseOptions.VarianceDecayRate = 1e-5;

agent = rlDDPGAgent(actor,critic,agentOptions);

maxepisodes = 5000; 
maxsteps = ceil(T/Ts);

trainingOpts = rlTrainingOptions(...
    'MaxEpisodes', maxepisodes, ...
    'MaxStepsPerEpisode', maxsteps, ...
    'Verbose', false, ...
    'Plots', 'training-progress', ...
    'StopTrainingCriteria', 'EpisodeReward', ...
    'StopTrainingValue', -1, ...
    'SaveAgentCriteria', 'EpisodeReward', ...
    'SaveAgentValue', -2.5);

%% 训练或加载预训练Agent

% 判断是否进行训练
doTraining = true;

if doTraining    
    % 训练Agent
    trainingStats = train(agent, env, trainingOpts);
else
    % 加载预训练Agent
    load('hil1.mat', 'agent')       
end

% 运行仿真
e1_initial = 0;
e2_initial = 0;
sim(mdl)

if ~doTraining
    % 关闭模型
    % bdclose(mdl)
end

%% 定义 Reset 和 Reward 函数

% Reset 函数
function in = localResetFcn(in)
    % 设置Memory模块的初始条件
    in = setVariable(in, 'Memory', [0, 0, 0]);
   
    for i = 1:3
        in = setVariable(in, ['e1_initial_', num2str(i)], 0); % 横向偏移
        in = setVariable(in, ['e2_initial_', num2str(i)], 0); % 相对偏航角
    end
end



