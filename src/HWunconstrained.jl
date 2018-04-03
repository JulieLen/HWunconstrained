

module HWunconstrained

	# imports: which packages are we going to use in this module?
	using Distributions, Optim, Plots, DataFrames

	"""
    `input(prompt::AbstractString="")`

    Read a string from STDIN. The trailing newline is stripped.

    The prompt string, if given, is printed to standard output without a
    trailing newline before reading input.
    """
    function input(prompt::AbstractString="")
        print(prompt)
        return chomp(readline())
    end

    export maximize_like_grad, runAll, makeData



	# methods/functions
	# -----------------

	# data creator
	# should/could return a dict with beta,numobs,X,y,norm)
	# true coeff vector, number of obs, data matrix X, response vector y, and a type of parametric distribution for G.
	function makeData(n=10_000)

	   N = n
	   beta = [1; 1.5; -0.5]
	   Data = DataFrame(rand(MvNormal([0,0,0.],[1 0 0; 0 1 0; 0 0 1]),N)')
	   Data[:x1]*beta[1] + Data[:x2]*beta[2] + Data[:x3]*beta[3]
	   Data[:P] = cdf.(Normal(), Data[:x1]*beta[1] + Data[:x2]*beta[2] + Data[:x3]*beta[3])
	   Data[:Dist] = Binomial.(1, Data[:P])
	   Data[:y] = rand.(Data[:Dist])

	   return Data

	end


	# log likelihood function at x
	# function loglik(betas::Vector,X::Matrix,y::Vector,distrib::UnivariateDistribution)
	function loglik(betas::Vector,d::DataFrame)

	   B = betas
	   P = cdf.(Normal(), d[:x1]*B[1] + d[:x2]*B[2] + d[:x3]*B[3])

	   el_loglik = d[:y] .* log.(P) + ( 1 - d[:y] ) .* log.(1-P)

	   return sum(el_loglik)

	end

	# gradient of the likelihood at x
	function grad!(storage::Vector,betas::Vector,d)
		xbeta     = d["X"]*betas	# (n,1)
		G_xbeta   = cdf.(d["dist"],xbeta)	# (n,1)
		g_xbeta   = pdf.(d["dist"],xbeta)	# (n,1)
		storage[:]= -mean((d["y"] .* g_xbeta ./ G_xbeta - (1-d["y"]) .* g_xbeta ./ (1-G_xbeta)) .* d["X"],1)
		return nothing
	end

	# hessian of the likelihood at x
	function hessian!(storage::Matrix,betas::Vector,d)
		return nothing
	end


	function info_mat(betas::Vector,d)
	end

	function inv_Info(betas::Vector,d)
	end



	"""
	inverse of observed information matrix
	"""
	function inv_observedInfo(betas::Vector,d)
	end

	"""
	standard errors
	"""
	function se(betas::Vector,d::Dict)
		# sqrt.(diag(inv_observedInfo(betas,d)))
		sqrt.(diag(inv_Info(betas,d)))
	end

	# function that maximizes the log likelihood without the gradient
	# with a call to `optimize` and returns the result
	function maximize_like(x0=[0.8,1.0,-0.1],meth=NelderMead())
		d = makeData(10000)
		res = optimize(arg->loglik(arg,d),x0,meth, Optim.Options(iterations = 500,g_tol=1e-20))
		return res
	end
	function maximize_like_helpNM(x0=[ 1; 1.5; -0.5 ],meth=NelderMead())
		d = makeData(10000)
		res = optimize(arg->loglik(arg,d),x0,meth, Optim.Options(iterations = 500,g_tol=1e-20))
		return res
	end



	# function that maximizes the log likelihood with the gradient
	# with a call to `optimize` and returns the result
	function maximize_like_grad(x0=[0.8,1.0,-0.1],meth=BFGS())
		return res
	end

	function maximize_like_grad_hess(x0=[0.8,1.0,-0.1],meth=Newton())
		return res
	end

	function maximize_like_grad_se(x0=[0.8,1.0,-0.1],meth=BFGS())
	end


	# visual diagnostics
	# ------------------

	# function that plots the likelihood
	# we are looking for a figure with 3 subplots, where each subplot
	# varies one of the parameters, holding the others fixed at the true value
	# we want to see whether there is a global minimum of the likelihood at the the true value.
	function plotlik()

	   eps = 1.5
	   beta1 = 1
	   beta2 = 1.5
	   beta3 = -0.5
	   r = 1000

	   # test for Beta 1

	   x1 = collect(linspace((beta1-eps), (beta1+eps),r))
	   y1 = []

	   for i in 1:r
	      push!(y1, loglik([x1[i], beta2, beta3], D))
	   end

	   P1 = plot(x1, y1, label="")
	   title!("Variations in beta1")
	   vline!([1.0], color=:grey,lw=1)

	   # test for Beta 2

	   x2 = collect(linspace((beta2-eps), (beta2+eps),r))
	   y2 = []

	   for i in 1:r
	      push!(y2, loglik([beta1, x2[i], beta3], D))
	   end

	   P2 = plot(x2, y2, label="")
	   title!("Variations in beta2")
	   vline!([1.5], color=:grey,lw=1)

	   # Test for Beta 3

	   x3 = collect(linspace((beta3-eps), (beta3+eps),r))
	   y3 = []

	   for i in 1:r
	      push!(y3, loglik([beta1, beta2, x3[i]], D))
	   end

	   P3 = plot(x3, y3, label="")
	   title!("Variations in beta3")
	   vline!([-0.5], color=:grey,lw=1)

	   plot(P1, P2, P3, label="")

	end

	function plotGrad()
		d = makeData(10000)

	end



	function runAll()

		plotLike()
		plotGrad()
		m1 = maximize_like()
		m2 = maximize_like_grad()
		m3 = maximize_like_grad_hess()
		m4 = maximize_like_grad_se()
		println("results are:")
		println("maximize_like optimizer: $(m1.minimizer)")
		println("maximize_like iterations: $(Optim.iterations(m1))")
		println("maximize_like_grad: $(Optim.minimizer(m2))")
		println("maximize_like_grad iterations: $(Optim.iterations(m2))")
		println("maximize_like_grad_hess: $(m3.minimizer)")
		println("maximize_like_grad_hess iterations: $(Optim.iterations(m3))")
		println("maximize_like_grad_se: $m4)")
		println("")
		println("running tests:")
		include("test/runtests.jl")
		println("")
		if isinteractive()
			ok = input("enter y to close this session.")
			if ok == "y"
				quit()
			end
		end
	end


end
