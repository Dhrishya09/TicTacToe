package ticTacToe;

import java.util.List;
import java.util.Random;

/**
 * A Q-Learning agent with a Q-Table, i.e. a table of Q-Values. This table is implemented in the {@link QTable} class.
 * 
 *  The methods to implement are: 
 * (1) {@link QLearningAgent#train}
 * (2) {@link QLearningAgent#extractPolicy}
 * 
 * Your agent acts in a {@link TTTEnvironment} which provides the method {@link TTTEnvironment#executeMove} which returns an {@link Outcome} object, in other words
 * an [s,a,r,s']: source state, action taken, reward received, and the target state after the opponent has played their move. You may want/need to edit
 * {@link TTTEnvironment} - but you probably won't need to. 
 * @author ae187
 */

public class QLearningAgent extends Agent {
	
	/**
	 * The learning rate, between 0 and 1.
	 */
	double alpha=0.1;
	
	/**
	 * The number of episodes to train for
	 */
	int numEpisodes=10000;
	
	/**
	 * The discount factor (gamma)
	 */
	double discount=0.9;
	
	
	/**
	 * The epsilon in the epsilon greedy policy used during training.
	 */
	double epsilon=0.1;
	
	/**
	 * This is the Q-Table. To get an value for an (s,a) pair, i.e. a (game, move) pair.
	 * 
	 */
	
	QTable qTable=new QTable();
	
	
	/**
	 * This is the Reinforcement Learning environment that this agent will interact with when it is training.
	 * By default, the opponent is the random agent which should make your q learning agent learn the same policy 
	 * as your value iteration and policy iteration agents.
	 */
	TTTEnvironment env=new TTTEnvironment();
	
	
	/**
	 * Construct a Q-Learning agent that learns from interactions with {@code opponent}.
	 * @param opponent the opponent agent that this Q-Learning agent will interact with to learn.
	 * @param learningRate This is the rate at which the agent learns. Alpha from your lectures.
	 * @param numEpisodes The number of episodes (games) to train for
	 */
	public QLearningAgent(Agent opponent, double learningRate, int numEpisodes, double discount)
	{
		env=new TTTEnvironment(opponent);
		this.alpha=learningRate;
		this.numEpisodes=numEpisodes;
		this.discount=discount;
		initQTable();
		train();
	}
	
	/**
	 * Initialises all valid q-values -- Q(g,m) -- to 0.
	 *  
	 */
	
	protected void initQTable()
	{
		List<Game> allGames=Game.generateAllValidGames('X');//all valid games where it is X's turn, or it's terminal.
		for(Game g: allGames)
		{
			List<Move> moves=g.getPossibleMoves();
			for(Move m: moves)
			{
				this.qTable.addQValue(g, m, 0.0);
				//System.out.println("initing q value. Game:"+g);
				//System.out.println("Move:"+m);
			}
			
		}
		
	}
	
	/**
	 * Uses default parameters for the opponent (a RandomAgent) and the learning rate (0.2). Use other constructor to set these manually.
	 */
	public QLearningAgent()
	{
		this(new RandomAgent(), 0.1, 69900, 0.9);
		
	}
	
	
	/**
	 *  Implement this method. It should play {@code this.numEpisodes} episodes of Tic-Tac-Toe with the TTTEnvironment, updating q-values according 
	 *  to the Q-Learning algorithm as required. The agent should play according to an epsilon-greedy policy where with the probability {@code epsilon} the
	 *  agent explores, and with probability {@code 1-epsilon}, it exploits. 
	 *  
	 *  At the end of this method you should always call the {@code extractPolicy()} method to extract the policy from the learned q-values. This is currently
	 *  done for you on the last line of the method.
	 */
	 
	 //helper function1 -> to find best move for current state
	     private Move BestMove(Game gameState, List<Move> moves) {
    	Move bestMove = null;
    	double maxQValue = -Double.MAX_VALUE;
        
        for (Move move : moves) {//iterate over all possible moves
            double qValue = qTable.getQValue(gameState, move);//get q value of current move
            if (qValue > maxQValue) {//compare qvalue of current move and max value, if greater
                maxQValue = qValue;//we update max value with current q value
                bestMove = move;//set this move as best move
            }
        }
        return bestMove;//after going through all possible moves , we return the highest q value move as best move
    }
	
    //helper function 2->Choose a move using an epsilon-greedy policy
    private Move Epsilon_greedyfun(Game gameState) {
        List<Move> possibleMoves = gameState.getPossibleMoves();//get all possible moves for the current state
        Random random = new Random();//intialise random no generator

        // If the random value is less than epsilon, explore by choosing a random move
        if (random.nextDouble() < epsilon) {
            return possibleMoves.get(random.nextInt(possibleMoves.size()));//chooses a random move by possibleMoves
        }
        return BestMove(gameState, possibleMoves);//if epilson value is higher then we choose the best move based on q value
    }
    
    //Helper function 3->Calculate maximum Q-value for a given game state
    private double MaxQValue(Game gameState) {
        if (gameState.isTerminal()) return 0.0;//q value is 0 for terminal state

        double maxQValue = -Double.MAX_VALUE;//initialise maximun q value
        for (Move move : gameState.getPossibleMoves()) {//iterate over all possible moves
            double qValue = qTable.getQValue(gameState, move);//get q value for current move
            if (qValue > maxQValue) {//compare to check if the q value of current move is higher than maxQValue
                maxQValue = qValue;//update with higher value
            }
        }
        return maxQValue;//return highest q value
    }
    
    //Helper function 4->Update Q-value using Q-Learning formula
	//formula= (1-a)Q+a(r+disc* max qvalue of next state)
    private void updateQValue(Outcome result) {
        double currentQValue = qTable.getQValue(result.s, result.move);//current q value
        double maxFutureQ = result.sPrime.isTerminal() ? 0 : MaxQValue(result.sPrime);//max q value for next state
        double updatedQValue = (1 - alpha) * currentQValue + alpha * (result.localReward + discount * maxFutureQ);//update q value using formula of q value

        qTable.addQValue(result.s, result.move, updatedQValue);//update q table with new value
    }
	
	public void train()
	{
		/* 
		 * YOUR CODE HERE
		 */
		for (int ep = 0; ep < numEpisodes; ep++) {//loop through each training episode
            while (!env.isTerminal()) {//loop goes on until we reach a terminal state
                Game game_state = env.getCurrentGameState();//get current
                if (game_state.isTerminal()) break;//if state is terminal,we break from the loop

                Move selected_move = Epsilon_greedyfun(game_state);//select move using epilson greedy move
                Outcome result = null;//initialise
                try {
                	result= env.executeMove(selected_move);//execute the move ,get the outcome and store in result
                } catch (IllegalMoveException e) {//to handle illegal moves (error)
                    e.printStackTrace();
                    continue;
                }

                updateQValue(result);//update q value
            }
            env.reset(); // Reset the environment for the next episode
        }
		
		//--------------------------------------------------------
		//you shouldn't need to delete the following lines of code.
		this.policy=extractPolicy();
		if (this.policy==null)
		{
			System.out.println("Unimplemented methods! First implement the train() & extractPolicy methods");
			//System.exit(1);
		}
	}
	
	/** Implement this method. It should use the q-values in the {@code qTable} to extract a policy and return it.
	 *
	 * @return the policy currently inherent in the QTable
	 */
	public Policy extractPolicy()
	{
		/* 
		 * YOUR CODE HERE
		 */
		
		Policy extractedPolicy = new Policy();//create new policy object

        for (Game game : qTable.keySet()) {//iterate all states in q table
            if (game.isTerminal()){//skip terminal states
				continue;
			};

            Move bestMove = BestMove(game, game.getPossibleMoves());//find best move for current state
            if (bestMove != null) {
                extractedPolicy.policy.put(game, bestMove);//add best move using policy
            }
        }
        return extractedPolicy;
		
	}
	
	public static void main(String a[]) throws IllegalMoveException
	{
		//Test method to play your agent against a human agent (yourself).
		QLearningAgent agent=new QLearningAgent();
		
		HumanAgent d=new HumanAgent();
		
		Game g=new Game(agent, d, d);
		g.playOut();
		
		
		

		
		
	}
	
	
	


	
}
