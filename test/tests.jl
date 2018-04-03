
@testset "basics" begin

	@testset "Test Data Construction" begin
		@test typeof(HWunconstrained.makeData()[:y]) == Array{Int64,1}
	end

	@testset "Test Return value of likelihood" begin

		@test HWunconstrained.loglik([0;0;0], HWunconstrained.makeData()) ≈ 10000*log(0.5)

	end

	@testset "Test return value of gradient" begin

	end
end

@testset "test maximization results" begin

	@testset "maximize returns approximate result" begin
	end

	@testset "maximize_grad returns accurate result" begin
	end

	@testset "maximize_grad_hess returns accurate result" begin
	end

	@testset "gradient is close to zero at max like estimate" begin

	end

end

@testset "test against GLM" begin

	@testset "estimates vs GLM" begin


	end

	@testset "standard errors vs GLM" begin


	end

end
